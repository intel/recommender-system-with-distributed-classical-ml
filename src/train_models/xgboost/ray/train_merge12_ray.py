import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
import yaml
import glob 

very_start = time.time()

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..','..','..'))
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

    train_path = list(sorted(glob.glob(f'{path}/stage2_train/*.parquet')))
    valid_path = list(sorted(glob.glob(f'{path}/stage2_valid/*.parquet')))
    pred_path = f"{pred_save_path}/xgboost_pred_stage1.csv"

    num_train = len(train_path) 
    num_valid = len(valid_path)

    start = time.time()
    df1 = pd.read_parquet(train_path)
    df2 = pd.read_parquet(valid_path)
    preds = pd.read_csv(pred_path)
    print('reading files took %.1f seconds'%(time.time()-start))

    start = time.time()
    index_cols = ['tweet_id', 'engaging_user_id']
    df1 = df1.merge(preds, on=index_cols, how="left")
    df2 = df2.merge(preds, on=index_cols, how="left")
    print('merging files took %.1f seconds'%(time.time()-start))

    # for col in df2.columns:
    #     if df2[col].dtype=='bool':
    #         df1[col] = df1[col].astype('int8')
    #         df2[col] = df2[col].astype('int8')

    # df1.to_parquet(f"{save_data_path}/stage2_train_pred.parquet")
    # df2.to_parquet(f"{save_data_path}/stage2_valid_pred.parquet")
    
    # start = time.time()
    # train_splits = np.array_split(df1, num_train)
    # valid_splits = np.array_split(df2, num_valid)
    # print('splitting files took %.1f seconds'%(time.time()-start))

    # start = time.time()
    # for i, df in enumerate(train_splits):
    #     df.to_parquet(f"{save_data_path}/stage2_train2/stage2_train_pred_{i}.parquet")
    
    # for i, df in enumerate(valid_splits):
    #     df.to_parquet(f"{save_data_path}/stage2_valid2/stage2_valid_pred_{i}.parquet")
    
    print('saving files took %.1f seconds'%(time.time()-start))

    print('This notebook took %.1f seconds'%(time.time()-very_start))
