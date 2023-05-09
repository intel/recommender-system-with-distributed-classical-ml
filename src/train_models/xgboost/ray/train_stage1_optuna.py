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
import optuna
from optuna import Trial, visualization
from optuna.samplers import TPESampler
import plotly
from optuna.pruners import MedianPruner

very_start = time.time()

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def read_data(save_data_path):
    # train = pd.read_parquet(f'{save_data_path}/train/stage1/train/')
    # valid = pd.read_parquet(f'{save_data_path}/train/stage1/valid/')
    train = pd.read_parquet(glob.glob(f'{save_data_path}/train/stage1/train/*.parquet')[0])
    valid = pd.read_parquet(glob.glob(f'{save_data_path}/train/stage1/valid/*.parquet')[0])

    print(train.shape)
    print(valid.shape)

    return (train, valid)

def train_xgboost(train_features, train_label, valid_features, valid_label, xgb_parms):
    dtrain = xgb.DMatrix(data=train_features, label=train_label)
    dvalid = xgb.DMatrix(data=valid_features, label=valid_label)

    model = xgb.train(xgb_parms, 
            dtrain=dtrain,
            evals=[(dtrain,'train'),(dvalid,'valid')],
            num_boost_round=500,
            early_stopping_rounds=25,
            #maximize=True,
            verbose_eval=25) 
    return model 

def save_prediction(oof, valid, save_path):
    for i in range(4):
        valid[f"pred_{label_names[i]}"] = oof[:,i]
    
    valid[["tweet_id","engaging_user_id",f"pred_{label_names[0]}",f"pred_{label_names[1]}",f"pred_{label_names[2]}",f"pred_{label_names[3]}"]].to_csv(save_path, index=0)
 

def evaluate_results(oof, valid):
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


label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']
feature_list = []
feature_list.append(stage1_reply_features[:-1])
feature_list.append(stage1_retweet_features[:-1])
feature_list.append(stage1_comment_features[:-1])
feature_list.append(stage1_like_features[:-1])

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..'))

try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

save_data_path = config['files']['save_data_path'] 
model_save_path = config['files']['model_save_path'] 
pred_save_path = config['files']['pred_save_path'] 

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


if __name__ == "__main__":
    ######## Load data
    train, valid = read_data(save_data_path)
    
    ####### Empty array to store prediction values
    oof = np.zeros((len(valid),len(label_names)))
    
    ######## Train and predict
    for numlabel in range(2,3):
        name = label_names[numlabel]
        print('#'*25);print('###',name);print('#'*25)
        print(f"Feature length for target {name} is {len(feature_list[numlabel])}")

        print(f"Label counts in training dataset: ")
        print(f"{pd.Series(train[name]).value_counts()}")

        dtrain = xgb.DMatrix(data=train[feature_list[numlabel]], label=train[name])
        dvalid = xgb.DMatrix(data=valid[feature_list[numlabel]], label=valid[name])

        def objective(trial, group=1):
            param = {
                        'objective':'binary:logistic',
                        'tree_method':'hist',
                        'eval_metric':'logloss',
                        'random_state': 42,
                        #'max_depth':trial.suggest_int('max_depth', 6, 15),
                        'max_depth': 8,
                        # 'reg_alpha':trial.suggest_uniform('reg_alpha',0,6),
                        # 'reg_lambda':trial.suggest_uniform('reg_lambda',0,2),
                        "reg_alpha":5.707312854,
                        "reg_lambda":1.217747347,
                        'min_child_weight':0.65155358,
                        #'min_child_weight':trial.suggest_loguniform('min_child_weight',0.5,2),
                        "gamma": 0,
                        #'gamma':trial.suggest_int('gamma', 0, 5),
                        #'eta':trial.suggest_loguniform('eta',0.01, 1),
                        'eta': 0.1,
                        'colsample_bytree': 0.85,
                        'subsample': 0.8,
                        
                        # 'colsample_bytree':trial.suggest_discrete_uniform('colsample_bytree',0.5, 1, 0.05),
                        # 'subsample':trial.suggest_discrete_uniform('subsample', 0.5, 1, 0.05),
                        "max_delta_step":1,
                        #"max_delta_step":trial.suggest_int('max_delta_step',1,10),
                        "scale_pos_weight": trial.suggest_loguniform('scale_pos_weight',0.5, 6),
                     }

            # param = {}
            
            # param['objective'] = 'binary:logistic'
            # param['tree_method'] = 'hist'
            # param['eval_metric'] = 'logloss'
            # param['random_state'] = 42

            # ## Initial Learning Parameters
            # param['learning_rate'] = 0.1
            #param['num_boost_round'] = 250
            
            # if group == 1:
            #     param['max_depth'] = trial.suggest_int('max_depth', 5, 12)
            #     param['min_child_weight'] = trial.suggest_loguniform('min_child_weight', 1e-10, 1e10)

            #     param['max_depth'] = 8
            #     param['subsample'] = 0.8
            #     param['colsample_bytree'] = 0.8

            # if group == 2:
            #     param['subsample'] = trial.suggest_uniform('subsample', 0, 1)
            #     param['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0, 1)
            
            # if group == 3:
            #     param['learning_rate'] = trial.suggest_uniform('learning_rate', 0, 0.1)
            #     param['num_boost_round'] = trial.suggest_int('num_boost_round', 100, 1000)


            #num_boost_round=trial.suggest_int('num_boost_round', 100, 500),

            model = xgb.train(param, 
                            dtrain=dtrain,
                            evals=[(dtrain,'train'),(dvalid,'valid')],
                            num_boost_round = 500,
                            early_stopping_rounds=25,
                            #maximize=True,
                            verbose_eval=25)
                            #callbacks=[optuna.integration.XGBoostPruningCallback(trial, "valid-logloss")]) 
            
            result = model.predict(dvalid)
            rce = compute_rce_fast(result,valid[label_names[numlabel]].values)
            ap = compute_AP(result,valid[label_names[numlabel]].values)
            
            return rce, ap  

        #model.save_model(f"{model_save_path}/xgboost_{name}_stage1.model")
        
        pruner = MedianPruner(n_warmup_steps=5)
        study1 = optuna.create_study(directions=['maximize','maximize'], sampler=TPESampler(), pruner=pruner)
        study1.optimize(objective, n_trials= 10, show_progress_bar = True)

        print('------------------------------------------------')
        print("Best trial:")
        best_trials = study1.best_trials

        print(len(best_trials))
        print("  Value: {}".format(best_trials[0].values))
        print("  Params: ")
        for key, value in best_trials[0].params.items():
            print("    {}: {}".format(key, value))

        # optuna.visualization.plot_optimization_history(study1).write_image(f"op_history_{name}_r2.png")
        # optuna.visualization.plot_slice(study1).write_image(f"slice_{name}_r2.png")
        # optuna.visualization.plot_param_importances(study1).write_image(f"param_importance_{name}_r2.png")

        # print('--------------------predict----------------------------')
        # best_model = xgb.train(best_trial.params, 
        #                 dtrain=dtrain,
        #                 evals=[(dtrain,'train'),(dvalid,'valid')],
        #                 num_boost_round = 500,
        #                 early_stopping_rounds=25,
        #                 #maximize=True,
        #                 verbose_eval=25)
        # oof[:,numlabel] = best_model.predict(dvalid)
        
    ######## Merge prediction to data and save
    #save_prediction(oof, valid, f"{pred_save_path}/xgboost_pred_stage1.csv")

    ######## Evaluate the performance
    #evaluate_results(oof, valid) 
    
    print('This notebook took %.1f seconds'%(time.time()-very_start))




