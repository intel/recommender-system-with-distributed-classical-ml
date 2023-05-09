#!/usr/bin/env python
# coding: utf-8
# this script is experimental, since it uses the newest pyspark wrapper API for xgboost4j-spark, which is not fully released 
import argparse
from pyspark import *
from pyspark.sql import *
from xgboost.spark import SparkXGBClassifier
from features import *
from pyspark.ml.feature import VectorAssembler
from pyspark.sql.functions import lit
from pathlib import Path
import yaml
import numpy as np 
from pyspark.ml.linalg import Vectors
import time

very_start = time.time()

xgb_parms = { 
    'learning_rate':0.1, 
    'max_depth':8, 
    'eval_metric':'logloss',
    'tree_method':'hist',
    'missing':0.0,
    'random_state':42,
    'subsample':0.8,
    'colsample_bytree':0.8, 
    'num_workers':16
}

num_rounds = [87, 213, 219, 18]

label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

feature_list = []
feature_list.append(stage1_reply_features[:-1])
feature_list.append(stage1_retweet_features[:-1])
feature_list.append(stage1_comment_features[:-1])
feature_list.append(stage1_like_features[:-1])


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
        .config("spark.driver.memory", '30g')\
        .config("spark.local.dir", "/mnt/tmp/spark")\
        .config("spark.executor.memory", "30g")\
        .config("spark.executor.memoryOverhead", "20g")\
        .config("spark.executor.cores", "8")\
        .config("spark.executor.instances","16")\
        .config("spark.task.cpus","8")\
        .getOrCreate()
            
    return spark


def read_data(data_path):

    train = spark.read.parquet(data_path+'train/').withColumn('isVal', lit(False))
    print(train.count(), len(train.columns))
    
    valid = spark.read.parquet(data_path+'valid/').withColumn('isVal', lit(True))
    print(valid.count(), len(valid.columns))
    
    data = train.union(valid) 

    print(data.count(), len(data.columns))

    return data


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
    is_local = config['training']['is_local']
    master = config['training']['master']
    path_prefix = f"hdfs://{master}:9000/"
    train_data_path = '/recsys2021/'
    DEBUG = config['training']['debug']

    if is_local:
        print("setting up local")
        spark = setup_local()
    else:
        print("setting up standalone")
        spark = setup_standalone(master)
    
    #TODO: read a few files from a list
    if DEBUG:
        data = read_data(path_prefix+train_data_path)
    else:
        data = read_data(path_prefix+train_data_path)
    
    # for evaluation
    #valid_df = spark.read.parquet(f'{save_data_path}/train/stage1/train_sample/')
    #oof = np.zeros((valid_df.count(), len(label_names)))

    for i in range(4):
        start = time.time()
        name = label_names[i]
        print('#'*25);print('###',name);print('#'*25)
        
        vector_assembler = VectorAssembler()\
                            .setInputCols(feature_list[i])\
                            .setOutputCol("features")

        data_trans = vector_assembler.setHandleInvalid("skip").transform(data)
        
        #print(data_trans.rdd.getNumPartitions())
        #data_trans = data_trans.repartition(16)
        #print(data_trans.rdd.getNumPartitions())
        # print(type(data_trans))

        #xgb_classifier = SparkXGBClassifier(**xgb_parms, n_estimators=num_rounds[i], feature_cols='features', label_col=name)
        
        xgb_classifier = SparkXGBClassifier(num_workers=1, max_depth=5, missing=0.0, force_repartition=True, label_col=name, validation_indicator_col='isVal', early_stopping_rounds=1, eval_metric='logloss')
        
        #print(xgb_classifier.extractParamMap())
        xgb_clf_model = xgb_classifier.fit(data_trans)
        #print(xgb_clf_model.explainParams())    
        #xgb_classifier.save(f"{model_save_path}/xgboost_{name}_stage1.model")

        #xgb_clf_model.transform(valid_trans).show()
        #valid_trans = vector_assembler.setHandleInvalid("keep").transform(valid_df)
        print(f'Train {name} took {time.time()-start} seconds')
     
    print('This notebook took %.1f seconds'%(time.time()-very_start))

                          
# def setup_local():

#     spark = SparkSession.builder.master('local[*]')\
#         .appName("Recsys2021_xgboost_train")\
#         .config("spark.driver.memory", '300g')\
#         .config("spark.local.dir", "/mnt/tmp/spark")\
#         .getOrCreate()
    
#     return spark

# def setup_standalone(master_name):

#     spark = SparkSession.builder.master(f'spark://{master_name}:7077')\
#         .appName("Recsys2021_xgboost_train")\
#         .config("spark.driver.memory", '30g')\
#         .config("spark.local.dir", "/mnt/tmp/spark")\
#         .config("spark.executor.memory", "30g")\
#         .config("spark.executor.memoryOverhead", "20g")\
#         .config("spark.executor.cores", "8")\
#         .config("spark.executor.instances","16")\
#         .config("spark.task.cpus","8")\
#         .getOrCreate()
            
#     return spark


#spark = setup_local()

# spark = setup_standalone("mlp-prod-icx-108398")

# df_train = spark.createDataFrame([
#             (Vectors.dense(1.0, 2.0, 3.0), 0, False, 1.0),
#             (Vectors.sparse(3, {1: 1.0, 2: 5.5}), 1, False, 2.0),
#             (Vectors.dense(4.0, 5.0, 6.0), 0, True, 1.0),
#             (Vectors.sparse(3, {1: 6.0, 2: 7.5}), 1, True, 2.0),
#             ], ["features", "label", "isVal", "weight"])

# print(type(df_train))

# print(df_train.rdd.getNumPartitions())

# df_test = spark.createDataFrame([
#             (Vectors.dense(1.0, 2.0, 3.0), ),
#             ], ["features"])


# xgb_classifier = SparkXGBClassifier(max_depth=5, missing=0.0,
#                                     validation_indicator_col='isVal',
#                                     early_stopping_rounds=1, eval_metric='logloss')

# xgb_clf_model = xgb_classifier.fit(df_train)
# xgb_clf_model.transform(df_test).show()
