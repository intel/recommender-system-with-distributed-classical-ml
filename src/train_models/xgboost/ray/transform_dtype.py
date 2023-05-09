import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
import yaml

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..'))
try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")

path = config['files']['data_path'] 
save_data_path = config['files']['save_data_path'] 
DEBUG = config['training']['debug']
pred_save_path = pred_save_path = config['files']['pred_save_path']

if DEBUG:
    pred_save_path = Path(os.path.join(pred_save_path, 'test'))


if __name__ == "__main__":

    ori_train_path = f"{path}/train/stage1/train"
    ori_valid_path = f"{path}/train/stage1/valid"
    tar_train_path = f"{save_data_path}/train/stage1/train"
    tar_valid_path = f"{save_data_path}/train/stage1/valid"

    path_mapping = {ori_train_path:tar_train_path, ori_valid_path:tar_valid_path}

    for ori_path, tar_path in path_mapping.items():
        file_list = os.listdir(ori_path)

        for file_name in file_list:

            if file_name.endswith("parquet"):
                df = pd.read_parquet(f"{ori_path}/{file_name}")

                for col in df.columns:
                    if df[col].dtype=='bool':
                        df[col] = df[col].astype('int8')
            
                df.to_parquet(f"{tar_path}/{file_name}")
                print(f"{file_name} is done!")
