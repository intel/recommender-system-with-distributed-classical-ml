import glob 
import pandas as pd 
import os, sys 
from xgboost_ray import RayDMatrix, RayParams, train, predict, RayFileType, RayShardingMode
from datetime import datetime
import subprocess

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
        self.model_type = model_spec['model_type']
        self.model_params = model_spec['model_params']
        self.training_params = model_spec['training_params']
        self.worker_ips = worker_ips
        self.ray_params = ray_params

    def process(self):

        if self.model_type == 'xgboost':
            print("preparing data for distributed xgboost training...")
            train_path, valid_path, test_path = self.prepare_data(self.data_path, self.data_format, self.ignore_cols, self.data_split)
            self.distribute_data(self.data_path, self.worker_ips)
            print("start xgboost model training...")
            self.model = XGBoostModel(train_path, valid_path, test_path, self.target_col, 
                                        self.model_params, self.training_params, self.ray_params).fit()
        else:
            raise NotImplementedError('currently only xgboost model is supported')
    
    def prepare_data(self, data_path, data_format, ignore_cols, data_split):

        if data_format == 'csv':
            df = read_csv_files(data_path, engine='pandas', ignore_cols=ignore_cols)
        else:
            print("other data format is not supported.")

        train = data_split['train']
        valid = data_split['valid']
        test = data_split['test']
        
        train_df = eval(train)
        valid_df = eval(valid)
        test_df = eval(test)
    
        divide_save_df(train_df, data_format, f"{data_path}/train", 100)
        divide_save_df(valid_df, data_format, f"{data_path}/valid", 100)
        divide_save_df(test_df, data_format, f"{data_path}/test", 100)

        train_path = list(sorted(glob.glob(f'{data_path}/train/*.{data_format}')))
        valid_path = list(sorted(glob.glob(f'{data_path}/valid/*.{data_format}')))
        test_path = list(sorted(glob.glob(f'{data_path}/test/*.{data_format}')))

        return train_path, valid_path, test_path 

    def distribute_data(self, data_path, worker_ips):

        for ip in worker_ips:
            for name in ['train', 'valid', 'test']:
                command = f"scp -r -o StrictHostKeyChecking=no {data_path}/{name} {ip}:{data_path}"
                subprocess.Popen(command.split(), stdout=subprocess.PIPE)

    def save_model(self, save_path):
        now = str(datetime.now().strftime("%Y-%m-%d+%H%M%S"))
        self.model.save_model(f"{save_path}/{self.model_type}_{now}.model")
        print(f"{self.model_type} is saved.")


class XGBoostModel:

    def __init__(self, train_path, valid_path, test_path, target_col, model_params, training_params, ray_params):
        self.train_path = train_path
        self.valid_path = valid_path 
        self.test_path = test_path 
        self.target_col = target_col
        self.model_params = model_params
        self.training_params = training_params
        self.ray_params = ray_params 
        try:
            self.epoch_log_interval = training_params['verbose_eval']
        except:
            self.epoch_log_interval = 25

    def fit(self):
        dtrain = RayDMatrix(data=self.train_path, label=self.target_col)
        dvalid = RayDMatrix(data=self.valid_path, label=self.target_col)
        dtest = RayDMatrix(data=self.test_path, label=self.target_col)

        watch_list = [(dtrain,'train'), (dvalid, 'valid'),(dtest, 'test')]

        model = train(self.model_params, **self.training_params, dtrain=dtrain, evals=watch_list, ray_params=RayParams(**self.ray_params))    

        return model


