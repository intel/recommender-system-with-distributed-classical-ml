import pandas as pd
import os
import numpy as np
from pathlib import Path
import time
import yaml
import glob 
import argparse 
from pyspark.sql.session import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import *

very_start = time.time()

label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

def setup_local():

    spark = SparkSession.builder.master('local[*]')\
        .appName("Recsys2021_xgboost_train")\
        .config("spark.driver.memory", '300g')\
        .config("spark.local.dir", "/mnt/tmp/spark")\
        .getOrCreate()
    
    return spark

def setup_standalone(master_name):

    spark = SparkSession.builder.master(f'spark://{master_name}:7077')\
        .appName("Recsys2021_xgboost_train")\
        .config("spark.driver.memory", '25g')\
        .config("spark.local.dir", "/mnt/tmp/spark")\
        .config("spark.executor.memory", "30g")\
        .config("spark.executor.memoryOverhead", "15g")\
        .config("spark.executor.cores", "8")\
        .config("spark.executor.instances","16")\
        .config("spark.task.cpus","8")\
        .config("spark.driver.maxResultSize", "10g")\
        .getOrCreate()

    # spark = SparkSession.builder.master(f'spark://{master_name}:7077')\
    #     .appName("Recsys2021_xgboost_train")\
    #     .config("spark.driver.memory", '8g')\
    #     .config("spark.local.dir", "/mnt/tmp/spark")\
    #     .config("spark.executor.memory", "10g")\
    #     .config("spark.executor.memoryOverhead", "6g")\
    #     .config("spark.executor.cores", "8")\
    #     .config("spark.executor.instances","4")\
    #     .config("spark.task.cpus","8")\
    #     .config("spark.driver.maxResultSize", "8g")\
    #     .getOrCreate()
    return spark

def read_data(data_path):

    train = spark.read.parquet(data_path+'/train')
    print(train.count(), len(train.columns))
    
    valid = spark.read.parquet(data_path+'/valid')
    print(valid.count(), len(valid.columns))
    
    return train, valid


if __name__ == "__main__":

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

    save_data_path = config['files']['save_data_path'] 
    model_save_path = config['files']['model_save_path'] 
    pred_save_path = config['files']['pred_save_path'] 
    is_local = config['env']['is_local']
    master = config['env']['master']
    path_prefix = f"hdfs://{master}:9000/"
    hdfs_data_path = config['files']['hdfs_data_path']
    hdfs_model_path = config['files']['hdfs_model_path']
    hdfs_pred_path = config['files']['hdfs_pred_path']

    if is_local:
        print("setting up local")
        spark = setup_local()
    else:
        print("setting up standalone")
        spark = setup_standalone(master)
    
    train_data, valid_data = read_data(path_prefix+hdfs_data_path+'/stage2')

    index_cols = ['tweet_id', 'engaging_user_id']
    pred = spark.read.options(header='True', inferSchema='True').csv(path_prefix + hdfs_pred_path + f"/xgboost_pred_stage1.csv")

    train_data = train_data.join(pred, on=index_cols, how='left')
    valid_data = valid_data.join(pred, on=index_cols, how='left')
    cols = train_data.columns
    for col_name in cols:
        if train_data.schema[col_name].dataType == BooleanType():
            train_data = train_data.withColumn(col_name, col(col_name).cast('integer'))
            valid_data = valid_data.withColumn(col_name, col(col_name).cast('integer'))

    train_data.repartition(1).write.mode("overwrite").parquet( path_prefix + f"{hdfs_data_path}/stage2_train_pred.parquet")
    valid_data.repartition(1).write.mode("overwrite").parquet( path_prefix + f"{hdfs_data_path}/stage2_valid_pred.parquet")

    print('This notebook took %.1f seconds'%(time.time()-very_start))
