import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
import yaml
import argparse

very_start = time.time()

if __name__ == "__main__":
    ######## Load data
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--config-dir",
            required=True,
            type=str,
            help="speficy the config path")
    args, _ = parser.parse_known_args()

    try: 
        with open(args.config_dir,'r') as file:
            config = yaml.safe_load(file)
    except Exception as e:
        print("Errors reading the config file.")

    data_path = config['files']['data_path'] 
    pred_save_path = config['files']['pred_save_path'] 

    df1 = pd.read_parquet(f"{data_path}/stage2_train/")
    df2 = pd.read_parquet(f"{data_path}/stage2_valid/")

    pred_path = f"{pred_save_path}/xgboost_pred_stage1.csv"
    preds = pd.read_csv(pred_path)

    index_cols = ['tweet_id', 'engaging_user_id']
    df1 = df1.merge(preds, on=index_cols, how="left")
    df2 = df2.merge(preds, on=index_cols, how="left")

    df1.to_parquet(f"{data_path}/stage2_train_pred.parquet")
    df2.to_parquet(f"{data_path}/stage2_valid_pred.parquet")

    print('This notebook took %.1f seconds'%(time.time()-very_start))
