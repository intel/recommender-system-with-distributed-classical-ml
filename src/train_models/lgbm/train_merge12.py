import pandas as pd
import numpy as np
import time
import os 
import yaml
from pathlib import Path
very_start = time.time()

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..','..', '..'))
try: 
    with open(os.path.join(ROOT_DIR,'config.yaml'),'r') as file:
        config = yaml.safe_load(file)
except Exception as e:
    print("Errors reading the config file.")


path = config['files']['data_path'] 
pred_save_path = config['files']['pred_save_path']
DEBUG = config['training']['debug']

if DEBUG:
    pred_save_path = Path(os.path.join(pred_save_path, 'test'))


if __name__ == "__main__":
    df1 = pd.read_parquet(f"{path}/train/stage2/train/")
    df2 = pd.read_parquet(f"{path}/train/stage2/valid/")

    pred_path = f"{pred_save_path}/lgbm_pred_stage1.csv"
    preds = pd.read_csv(pred_path)

    index_cols = ['tweet_id', 'engaging_user_id']
    df1 = df1.merge(preds, on=index_cols, how="left")
    df2 = df2.merge(preds, on=index_cols, how="left")

    df1.to_parquet(f"{path}/stage2_train_pred.parquet")
    df2.to_parquet(f"{path}/stage2_valid_pred.parquet")

    print('This notebook took %.1f seconds'%(time.time()-very_start))
