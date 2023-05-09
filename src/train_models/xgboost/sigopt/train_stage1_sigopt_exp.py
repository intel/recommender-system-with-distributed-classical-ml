#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
import xgboost as xgb 
from sklearn.metrics import log_loss, average_precision_score
from features import *
import yaml
from pathlib import Path
import shutil
import os
import sys
from sigopt import Connection
import sigopt


api_token = 'EJNLQYETVCMUQVFAZJSWUQUAWJUNFSYVJMQLYTHQNEWGOTOL'
conn = Connection(client_token=api_token)

very_start = time.time()

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def read_data(save_data_path):
    # train = pd.read_parquet(f'{save_data_path}/train/stage1/train_sample2/')
    # valid = pd.read_parquet(f'{save_data_path}/train/stage1/valid_sample/')
    train = pd.read_parquet(f'{save_data_path}/train/stage1/train/')
    valid = pd.read_parquet(f'{save_data_path}/train/stage1/valid/')  

    print(train.shape)
    print(valid.shape)

    return (train, valid)

def save_prediction(oof, valid, save_path):
    for i in range(4):
        valid[f"pred_{label_names[i]}"] = oof[:,i]
    valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}",f"pred_{label_names[1]}",f"pred_{label_names[2]}",f"pred_{label_names[3]}"]].to_csv(save_path, index=0)
 
def evaluate_results(oof, valid):
    print('#'*25);print('###','Evalution Results');print('#'*25)
    txts = ''
    sumap = 0
    sumrce = 0
    for i in range(1):
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


label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
feature_list = []
feature_list.append(stage1_reply_features[:-1])
feature_list.append(stage1_retweet_features[:-1])
feature_list.append(stage1_comment_features[:-1])
feature_list.append(stage1_like_features[:-1])

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__),'..','..','..','..'))

try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

save_data_path = config['files']['save_data_path'] 
model_save_path = config['files']['model_save_path'] 
pred_save_path = config['files']['pred_save_path'] 


if __name__ == "__main__":
    ######## Load data
    train, valid = read_data(save_data_path)
    print("data loaded!")

    ####### Empty array to store prediction values
    oof = np.zeros((len(valid),len(label_names)))
    
    ######## Train and predict
    for numlabel in range(1):
        start = time.time()
        name = label_names[numlabel]
        
        print('#'*25);print('###',name);print('#'*25)
        print(f"Feature length for target {name} is {len(feature_list[numlabel])}")

        print(f"Label counts in training dataset: ")
        print(f"{pd.Series(train[name]).value_counts()}")

        weights = (train[name] == 0).sum() / (1.0 * (train[name] == 1).sum())
        # dtrain = xgb.DMatrix(data=train[feature_list[numlabel]], label=train[name])
        # dvalid = xgb.DMatrix(data=valid[feature_list[numlabel]], label=valid[name])
        
        eval_set = [(train[feature_list[numlabel]],train[name]),(valid[feature_list[numlabel]],valid[name])]
        def create_model(assignments):
            model = xgb.XGBClassifier(
                objective        = 'binary:logistic',
                tree_method      = 'hist',
                eval_metric      = 'logloss',
                random_state     = 42,
                scale_pos_weight = weights,
                
                #min_child_weight = assignments['min_child_weight'],
                max_depth        = assignments['max_depth'],
                learning_rate    = assignments['log_learning_rate'],
                subsample        = assignments['subsample'],
                colsample_bytree = assignments['colsample_bytree'],
                gamma            = assignments['gamma'],
                reg_alpha        = assignments['alpha'],
                reg_lambda       = assignments['lambda'],
                # SigOpt-optimized parameters end here
                )
            return model

        def evaluate_model(assignments):
            model = create_model(assignments)
            probabilities = model.fit(train[feature_list[numlabel]], train[name], early_stopping_rounds=25, eval_set = eval_set).predict_proba(valid[feature_list[numlabel]])
            #rce = compute_rce_fast(probabilities[:, 1], valid[label_names[numlabel]].values)
            ap = compute_AP(probabilities[:, 1], valid[label_names[numlabel]].values)
            return ap


        experiment = conn.experiments().create(
            name="RecSys Prediction XGB - Vanilla SigOpt 3",
            parameters=[
                #dict(name="min_child_weight", bounds=dict(min=np.log(0.5),max=np.log(2)), type="double"),
                dict(name="max_depth", bounds=dict(min=5,max=12), type="int"),
                dict(name="log_learning_rate", bounds=dict(min=0.01,max=1), type="double"),
                dict(name="subsample", bounds=dict(min=0.5, max=1), type="double"),
                dict(name="colsample_bytree", bounds=dict(min=0.5, max=1), type="double"),
                dict(name="gamma", bounds=dict(min=0,max=5), type="double"),
                dict(name="alpha", bounds=dict(min=0, max=6), type="double"),
                dict(name="lambda", bounds=dict(min=0, max=2), type="double")
                ],
            metrics=[
                dict(name="precision", objective="maximize", strategy="optimize")
                ],
            observation_budget = 30,
        )
        print("Explore your experiment: https://app.sigopt.com/experiment/" + experiment.id + "/analysis")


        for _ in range(experiment.observation_budget):
            suggestion = conn.experiments(experiment.id).suggestions().create()
            assignments = suggestion.assignments
            value = evaluate_model(assignments)
            conn.experiments(experiment.id).observations().create(
                suggestion=suggestion.id,
                value=value
            )
            #update experiment object
            experiment = conn.experiments(experiment.id).fetch()
        assignments = conn.experiments(experiment.id).best_assignments().fetch().data[0].assignments  
        print("BEST ASSIGNMENTS \n", assignments)

    ######## Evaluate the performance
    #evaluate_results(oof, valid) 
    
    print('This notebook took %.1f seconds'%(time.time()-very_start))




