import glob 
import pandas as pd 
import os 
import xgboost as xgb  
from datetime import datetime
import sys 

SRC_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, f"{SRC_PATH}/utils")

from data_utils import *


class Trainer:
    def __init__(self, data_path, data_format, data_spec, model_spec, worker_ips, ray_params):
        self.data_path = data_path
        self.data_format = data_format
        self.target_col = data_spec['target_col']
        try:
            self.ignore_cols = data_spec['ignore_cols']
        except:
            self.ignore_cols = None
        self.data_split = data_spec['data_split']
        self.has_test = True if len(self.data_split)==3 else False   
        self.model_type = model_spec['model_type']
        self.training_params = model_spec['training_params']
        self.model_params = model_spec['model_params']

    def process(self):
        print("read and prepare data for training...")
        train_df, valid_df, test_df = self.read_and_split(self.data_split, self.data_path, self.data_format, self.ignore_cols)
        
        if self.model_type == 'xgboost':
            print("start xgboost model training...")
            self.model = XGBoostModel(train_df, valid_df, test_df, self.target_col, 
                                        self.model_params, self.training_params).fit()
        else:
            raise NotImplementedError('currently only xgboost model is supported')
    

    def read_and_split(self, data_split, data_path, data_format, ignore_cols):

        train = data_split['train']
        valid = data_split['valid']
        test = data_split['test']

        if os.path.exists(train) and os.path.exists(valid):
            train_path = train
            valid_path = valid
            train_df = self.read_data(train_path, data_format, ignore_cols)
            valid_df = self.read_data(valid_path, data_format, ignore_cols)
            test_df = self.read_data(test, data_format, ignore_cols)
        else: 
            data_path = self.data_path
            df = self.read_data(data_path, data_format, ignore_cols)
            train_df = eval(train)
            valid_df = eval(valid)
            test_df = eval(test)

        return train_df, valid_df, test_df 


    def read_data(self, train_data_path, train_data_format, ignore_cols):        
        if train_data_format == 'csv':
            data = read_csv_files(train_data_path, engine='pandas', ignore_cols=ignore_cols)
        else: 
            raise NotImplementedError('currently only csv format is supported')
        
        return data 


    def save_model(self, save_path):
        now = str(datetime.now().strftime("%Y-%m-%d+%H%M%S"))
        self.model.save_model(f"{save_path}/{self.model_type}_{now}.model")
        print(f"{self.model_type} is saved.")


class XGBoostModel:

    def __init__(self, train_df, valid_df, test_df, target_col, model_params, training_params):
        self.train_df = train_df
        self.valid_df = valid_df 
        self.test_df = test_df 
        self.target_col = target_col
        self.model_params = model_params
        self.training_params = training_params
        try:
            self.epoch_log_interval = training_params['verbose_eval']
        except:
            self.epoch_log_interval = 25

    def fit(self):
        dtrain, dvalid, dtest = self.data_prep(self.train_df, self.valid_df, self.test_df, self.target_col)
        
        watch_list = [(dtrain,'train'), (dvalid, 'valid'),(dtest, 'test')]

        model = xgb.train(self.model_params, **self.training_params, dtrain=dtrain, evals=watch_list)    

        return model


    def data_prep(self, train_df, valid_df, test_df, label_col):
        
        dtrain = xgb.DMatrix(data=train_df.drop(label_col, axis=1), label=train_df[label_col])
        dvalid = xgb.DMatrix(data=valid_df.drop(label_col, axis=1), label=valid_df[label_col])
        dtest = xgb.DMatrix(data=test_df.drop(label_col, axis=1), label=test_df[label_col])

        return (dtrain, dvalid, dtest)

