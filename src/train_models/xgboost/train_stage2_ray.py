#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
from xgboost_ray import RayDMatrix, RayParams, train, predict, RayFileType
from sklearn.metrics import log_loss, average_precision_score
from features import *
import yaml
from pathlib import Path
import ray 


ray.init(address='auto')
cpus_per_actor = 15
num_actors = 20
ray_params = RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor, elastic_training=True, max_failed_actors=1, max_actor_restarts=1)


very_start = time.time()

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..'))
try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

save_data_path = config['files']['save_data_path'] 
DEBUG = config['training']['debug']
model_save_path = config['files']['model_save_path'] 

if __name__ == "__main__":
    ######## Load data
    train_path = list(sorted(glob.glob(f'{save_data_path}/train/stage2/train/*.parquet')))
    valid_path = list(sorted(glob.glob(f'{save_data_path}/train/stage2/valid/*.parquet')))
    
    valid = pd.read_parquet(f'{save_data_path}/train/stage2/valid/')    

    if DEBUG:
        model_save_path = f"{model_save_path}/test/"
        train_path = list(sorted(glob.glob(f'{save_data_path}/train/stage2/train/stage2_train_pred_0.parquet')))
        valid_path = list(sorted(glob.glob(f'{save_data_path}/train/stage2/valid/stage2_valid_pred_0.parquet')))
        valid = pd.read_parquet(f'{save_data_path}/train/stage2/valid/stage2_valid_pred_0.parquet')  

    print(valid.shape)

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list = []
    feature_list.append(stage2_reply_features)
    feature_list.append(stage2_retweet_features)
    feature_list.append(stage2_comment_features)
    feature_list.append(stage2_like_features)
    for i in range(4):
        print(len(feature_list[i])-1)

    ######## Train and predict
    xgb_parms = { 
        'max_depth':8, 
        'learning_rate':0.1, 
        'subsample':0.8,
        'colsample_bytree':0.8, 
        'eval_metric':'logloss',
        'objective':'binary:logistic',
        'tree_method':'hist',
        "random_state":42
    }

    oof = np.zeros((len(valid),len(label_names)))
    for numlabel in range(4):
        start = time.time()
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)


        dtrain = RayDMatrix(
                        train_path,
                        label=name,  # Will select this column as the label
                        columns=feature_list[numlabel],
                        # ignore=["total_amount"],  # Optional list of columns to ignore
                        filetype=RayFileType.PARQUET)
        
        dvalid = RayDMatrix(
                        valid_path,
                        label=name,  # Will select this column as the label
                        columns=feature_list[numlabel],
                        # ignore=["total_amount"],  # Optional list of columns to ignore
                        filetype=RayFileType.PARQUET)

        print("prepare matrix took %.1f seconds" % ((time.time()-start)))

        start = time.time()
        print("Training.....")
        model = train(xgb_parms, 
                dtrain=dtrain,
                evals=[(dtrain,'train'),(dvalid,'valid')],
                num_boost_round=250,
                early_stopping_rounds=25,
                #maximize=True,
                verbose_eval=25,
                ray_params=ray_params)
        print("training took %.1f seconds" % ((time.time()-start))) 
        model.save_model(f"{model_save_path}/xgboost_{name}_stage2.model")

        print('Predicting...')
        dvalid = RayDMatrix(
                valid_path,
                label=name,  # Will select this column as the label
                columns=feature_list[numlabel],
                # ignore=["total_amount"],  # Optional list of columns to ignore
                filetype=RayFileType.PARQUET)


        start = time.time()
        oof[:, numlabel] = predict(model, dvalid,  ray_params=RayParams(num_actors=1))
        print("prediction took %.1f seconds" % ((time.time()-start)))

    ######## Evaluate the performance
    print('#'*25);print('###','Evalution Results');print('#'*25)
    txts = ''
    sumap = 0
    sumrce = 0
    for i in range(4):
        ap = compute_AP(oof[:,i],valid[label_names[i]].values)
        rce = compute_rce_fast(oof[:,i],valid[label_names[i]].values)
        txt = f"{label_names[i]:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt)

        txts += "%.4f" % ap + ' '
        txts += "%.4f" % rce + ' '
        sumap += ap
        sumrce += rce
    print(txts)
    print("AVG AP: ", sumap/4.)
    print("AVG RCE: ", sumrce/4.)
  
    print('This notebook took %.1f seconds'%(time.time()-very_start))