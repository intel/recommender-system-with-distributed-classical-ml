#!/usr/bin/env python
# coding: utf-8
import argparse
import os, time, gc, sys, glob
import pandas as pd 
import numpy as np
import yaml 
import time 
from src.utils.data_utils import *

very_start = time.time()

PATH_HOME = os.path.dirname(os.path.realpath(__file__))

class WFProcessor:

    def __init__(self, file_name): 
        
        with open(os.path.join('/workspace/configs',os.path.basename(file_name)),'r') as file:
            config = yaml.safe_load(file)

        self.num_node = config['env']['num_node'] 
        self.is_multi_nodes = True if self.num_node > 1 else False  
        self.worker_ips = config['env']['node_ips'][1:]
        try:
            self.raw_data_path = os.path.join(os.path.dirname(PATH_HOME), 'data', config['data_preprocess']['input_data_path']) 
            self.raw_data_format = config['data_preprocess']['input_data_format'] 
            dp_config_file = os.path.join('/workspace/configs', config['data_preprocess']['dp_config_file']) 
            self.dp_framework = config['data_preprocess']['dp_framework']
            self.processed_data_path = os.path.join(os.path.dirname(PATH_HOME), 'data', config['data_preprocess']['output_data_path']) 
            self.processed_data_format = config['data_preprocess']['output_data_format']
            self.read_data_processing_steps(dp_config_file)
            self.has_dp = True
        except Exception as e: 
            print('Failed to read data preprocessing steps. This is either due to wrong parameters defined in the config file as shown: '+ str(e) )
            print("Or there is no need for data preprocessing.")
            self.has_dp = False
        try:
            self.train_data_path = os.path.join(os.path.dirname(PATH_HOME), 'data', config['training']['input_data_path']) 
            self.train_data_format = config['training']['input_data_format']
            self.train_model_path = '/workspace/tmp/models'
            train_config_file = os.path.join('/workspace/configs', config['training']['train_config_file']) 
            self.train_framework = config['training']['train_framework']
            self.read_training_configs(train_config_file)
            try:
                self.ray_params = config['training']['ray_params']
            except:
                self.ray_params = None 
            self.has_training = True
        except Exception as e: 
            print('Failed to read model training configurations. This is either due to wrong parameters defined in the config file as shown: '+ str(e) )
            print("Or there is no need for model training.")
            self.has_training = False
        self.cluster_engine = None

    def prepare_env(self):
        if self.is_multi_nodes: 
            print("enter distributed mode...")
            if self.has_dp and self.has_training:
                if self.dp_framework == 'spark' and self.train_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            elif self.has_dp and not self.has_training: 
                if self.dp_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            elif not self.has_dp and self.has_training: 
                if self.train_framework == 'spark':
                    self.cluster_engine='spark'
                else:
                    self.cluster_engine='ray'
            else:
                print("Check your workflow config file. Program End.")
                sys.exit()
            if self.cluster_engine == 'ray':
                print("initializing ray cluster...")
                import ray 
                ray.init('auto', runtime_env={'env_vars': {'__MODIN_AUTOIMPORT_PANDAS__': '1'}}, log_to_driver=False)
            if self.cluster_engine == 'spark':
                print("initializing spark cluster...")
                raise NotImplementedError
                
        else:
            print("enter single-node mode...")
            if not self.has_dp and not self.has_training:
                print("Program End.")
                sys.exit()
            
    def read_data_processing_steps(self, dp_config_file):
        with open(dp_config_file, 'r') as file:
            dp_steps = yaml.safe_load(file)
        
        self.data_preparation_steps = dp_steps['data_preparation']
        self.feature_engineer_steps = dp_steps['feature_engineering']
        self.feature_encoding_steps = dp_steps['feature_encoding']
        self.data_splitting_rule = dp_steps['data_splitting']

    def read_training_configs(self, train_config_file):
        with open(train_config_file, 'r') as file:
            train_configs = yaml.safe_load(file)
        
        self.train_data_spec = train_configs['data_spec']
        self.train_model_spec = train_configs['model_spec']
             
    def read_data(self):
        print('reading data...')
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            engine = 'pandas'
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            engine = 'modin'
        else:
            raise NotImplementedError('currently only pandas data preprocessing is supported')
        if self.raw_data_format == 'csv':
            self.data = read_csv_files(self.raw_data_path, engine=engine)

    def prepare_data(self):
        print('preparing data...')
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            from src.data_preprocess.pandas.data_preparation import DataPreparator
            data_preparator = DataPreparator(self.data, self.data_preparation_steps)
            self.data = data_preparator.process()
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            from src.data_preprocess.modin.data_preparation import DataPreparator
            data_preparator = DataPreparator(self.data, self.data_preparation_steps)
            self.data = data_preparator.process()
        else:
            raise NotImplementedError('currently only pandas data preprocessing is supported')
    
    def engineer_features(self):
        print('engineering features...')
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            from src.data_preprocess.pandas.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer(self.data, self.feature_engineer_steps)
            self.data = feature_engineer.process()
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            from src.data_preprocess.modin.feature_engineering import FeatureEngineer
            feature_engineer = FeatureEngineer(self.data, self.feature_engineer_steps)
            self.data = feature_engineer.process()
        else:
            raise NotImplementedError('currently only pandas data preprocessing is supported')

    def split_data(self):
        print('splitting data...')
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            from src.data_preprocess.pandas.data_splitting import DataSplitter 
            data_splitter = DataSplitter(self.data, self.data_splitting_rule)
            self.train_data, self.test_data = data_splitter.process()
            self.data = None 
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            from src.data_preprocess.modin.data_splitting import DataSplitter 
            data_splitter = DataSplitter(self.data, self.data_splitting_rule)
            self.train_data, self.test_data = data_splitter.process()
            self.data = None 
        else: 
            raise NotImplementedError('currently only pandas data preprocessing is supported')

    def encode_features(self):
        print('encoding features...')
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            from src.data_preprocess.pandas.feature_encoding import FeatureEncoder
            feature_encoder = FeatureEncoder(self.train_data, self.test_data, self.feature_encoding_steps)
            self.train_data, self.test_data = feature_encoder.process()
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            from src.data_preprocess.modin.feature_encoding import FeatureEncoder
            feature_encoder = FeatureEncoder(self.train_data, self.test_data, self.feature_encoding_steps)
            self.train_data, self.test_data = feature_encoder.process()
        else:
            raise NotImplementedError('currently only pandas data preprocessing is supported')

    def save_data(self, merge=False):
        print('saving data...')
        if self.dp_framework == 'pandas' and self.is_multi_nodes == False:
            import pandas as pd 
        elif self.dp_framework == 'pandas' and self.is_multi_nodes == True:
            import modin.pandas as pd 
        else:
            raise NotImplementedError('currently only pandas data preprocessing is supported')
        
        if self.processed_data_format == 'csv':
            data = pd.concat([self.train_data, self.test_data])
            data.to_csv(self.processed_data_path+'/processed_data.csv', index=False)
            print(f'data saved under the path {self.processed_data_path}/processed_data.csv')
    
    def train_model(self):
        print('start training models soon...')
        if self.train_framework == 'pandas' and self.is_multi_nodes == False:
            from src.train_models.xgboost.model_training import Trainer
        elif self.train_framework == 'pandas' and self.is_multi_nodes == True:
            from src.train_models.xgboost_ray.model_training import Trainer 
        else:
            raise NotImplementedError('currently only singe-node xgboost training is supported')
        
        trainer = Trainer(self.train_data_path, self.train_data_format, self.train_data_spec, self.train_model_spec, self.worker_ips, self.ray_params)
        trainer.process()
        trainer.save_model(self.train_model_path)
    
    def clean_up(self):
        if self.cluster_engine is None:
            pass 
        elif self.cluster_engine == 'ray':
            import ray 
            ray.shutdown()
        else:
            raise NotImplementedError('spark engine is to be implemented!')



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config-file",
            required=True,
            type=str,
            help="speficy the config file name")
    args, _ = parser.parse_known_args()
    
    wf_processor = WFProcessor(args.config_file)
    
    start = time.time()
    wf_processor.prepare_env()
    print("prepare env took %.1f seconds" % ((time.time()-start)))

    if wf_processor.has_dp:
        dp_start = time.time()
        start = time.time()
        wf_processor.read_data()
        print("dp read data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.prepare_data()
        print("dp prepare data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.engineer_features()
        print("dp engineer features took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.split_data()
        print("dp split data took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.encode_features()
        print("dp encode features took %.1f seconds" % ((time.time()-start)))
        start = time.time()
        wf_processor.save_data()
        print("dp save data took %.1f seconds" % ((time.time()-start)))
        print("data preprocessing took %.1f seconds" % ((time.time()-dp_start)))
    if wf_processor.has_training:
        train_start = time.time()
        wf_processor.train_model()
        print("training took %.1f seconds" % ((time.time()-train_start)))

    wf_processor.clean_up()
    print('The whole workflow processing took %.1f seconds'%(time.time()-very_start))
