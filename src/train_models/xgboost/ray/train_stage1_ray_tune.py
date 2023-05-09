#!/usr/bin/env python
# coding: utf-8
import os, time, gc, sys, glob
import pandas as pd
import numpy as np
from xgboost_ray import RayDMatrix, RayParams, train, predict, RayFileType
import xgboost_ray
from sklearn.metrics import log_loss, average_precision_score
from features import *
import yaml
from pathlib import Path
import shutil
import os
import sys
import ray 
from ray import tune 
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.integration.xgboost import TuneReportCheckpointCallback

very_start = time.time()

ray.init(address="auto")
num_actors = 20
cpus_per_actor = 15
num_samples = 20
ray_params = RayParams(num_actors=num_actors, cpus_per_actor=cpus_per_actor, elastic_training=True, max_failed_actors=1, max_actor_restarts=1)

space = {
    'objective':'binary:logistic',
    'tree_method':'hist',
    'eval_metric':"logloss",
    "eta": 0.1,
    "max_depth":8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "scale_pos_weight": tune.loguniform(0.5,2),
    "max_delta_step":tune.randint(1, 10)
    #"eta": tune.loguniform(1e-2, 1e-1),
    #"max_depth": tune.randint(5, 12),
    #"subsample": tune.uniform(0.5, 1.0),
    #"colsample_bytree": tune.uniform(0.5, 1.0)
    #"min_child_weight": tune.randint(1, 6),
    #"colsample_bytree": tune.uniform(0.5, 1.0)
}

optuna_search = OptunaSearch(metric="eval-logloss", mode="min")

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

def check_test_path(model_save_path):
    if model_save_path.is_dir():
        print('WARNING: test model directory \'{}\' already exists - will be removed and newly created '.format(
                    model_save_path), sys.stderr)
        shutil.rmtree(model_save_path, ignore_errors=True)
        model_save_path.mkdir(parents=True)
    else:
        print(
            'WARNING: test model directory \'{}\' did not exist - will be newly created '.format(
                model_save_path), sys.stderr)
        model_save_path.mkdir(parents=True)
    return None

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..'))

try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

save_data_path = config['files']['save_data_path'] 
model_save_path = config['files']['model_save_path'] 
pred_save_path = config['files']['pred_save_path'] 
num_iterations = config['training']['num_iterations']
DEBUG = config['training']['debug']

best_configs = []
if __name__ == "__main__":
    ######## Load data
    if DEBUG:
        num_iterations = config['training']['num_iterations_debug'] 
        model_save_path = Path(os.path.join(model_save_path, 'test'))
        pred_save_path = Path(os.path.join(pred_save_path, 'test'))

        check_test_path(model_save_path)
        check_test_path(pred_save_path)

        train_path = list(sorted(glob.glob(f'{save_data_path}/train/stage1/train/part-000**-edec5252-abaf-409b-9cc3-c024cd16da72-c000.snappy.parquet')))
        valid_path = list(sorted(glob.glob(f'{save_data_path}/train/stage1/valid/part-000**-5920f744-7273-4c28-a5f9-87ec5fff5f35-c000.snappy.parquet')))
        print(len(train_path))
        print(len(valid_path))
        valid = pd.read_parquet(glob.glob(f'{save_data_path}/train/stage1/valid/part-000**-5920f744-7273-4c28-a5f9-87ec5fff5f35-c000.snappy.parquet')) 
    else:
        train_path = list(sorted(glob.glob(f'{save_data_path}/train/stage1/train/*.parquet'))) # 200 files
        valid_path = list(sorted(glob.glob(f'{save_data_path}/train/stage1/valid/*.parquet'))) # 100 files 
        valid = pd.read_parquet(f'{save_data_path}/train/stage1/valid/') # load it using pandas for evaluation later
    print(valid.shape)

    ######## Feature list for each target
    label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
    feature_list = []
    feature_list.append(stage1_reply_features)
    feature_list.append(stage1_retweet_features)
    feature_list.append(stage1_comment_features)
    feature_list.append(stage1_like_features)
    for i in range(4):
        print(len(feature_list[i])-1)

    oof = np.zeros((len(valid),len(label_names)))

    for numlabel in range(4):
        start = time.time()
        
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)
        
        def train_model(config, ray_params):
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

            evals_result = {}
            bst = train(
                params=config,
                dtrain=dtrain,
                evals_result=evals_result,
                evals=[(dvalid, "eval")],
                verbose_eval=False,
                num_boost_round=1000,
                early_stopping_rounds=25,
                callbacks=[TuneReportCheckpointCallback(filename="model.xgb")],
                ray_params=ray_params)
            
            bst.save_model("model.xgb")
            print("Final validation error: {:.4f}".format(evals_result["eval"]["logloss"][-1]))
        
        analysis = tune.run(
                tune.with_parameters(train_model, ray_params=ray_params),
                config=space,
                search_alg=optuna_search,
                num_samples=num_samples,
                metric="eval-logloss", 
                mode="min",
                local_dir = '/mnt/data/ray',
                resources_per_trial=ray_params.get_tune_resources())
        
        # Load the best model checkpoint.
        best_bst = xgboost_ray.tune.load_model(os.path.join(analysis.best_logdir, "model.xgb"))
        #best_bst.save_model(f"{model_save_path}/xgboost_{name}_stage1.model")
        #accuracy = 1. - analysis.best_result["eval-logloss"]
        best_configs.append(analysis.best_config)
        print(f"Best model parameters: {analysis.best_config}")
        #print(f"Best model total accuracy: {accuracy:.4f}")

        print('Predicting...')        
        dvalid = RayDMatrix(
                        valid_path,
                        label=name,  # Will select this column as the label
                        columns=feature_list[numlabel],
                        # ignore=["total_amount"],  # Optional list of columns to ignore
                        filetype=RayFileType.PARQUET)
        
        start = time.time()
        oof[:, numlabel] = predict(best_bst, dvalid,  ray_params=RayParams(num_actors=1, cpus_per_actor=1))
        print("prediction took %.1f seconds" % ((time.time()-start)))

    ######## Merge prediction to data and save
    for i in range(4):
        valid[f"pred_{label_names[i]}"] = oof[:,i]
    
    #valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}",f"pred_{label_names[1]}",f"pred_{label_names[2]}",f"pred_{label_names[3]}"]].to_csv(f"{pred_save_path}/xgboost_pred_stage1.csv",index=0)
    #valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}"]].to_csv(f"{pred_save_path}/xgboost_pred_stage1.csv",index=0)
   
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
    