#!/usr/bin/env python
# coding: utf-8
from sparkxgb import XGBoostClassifier, XGBoostRegressor
from features import *
import pyspark
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col, udf
from pyspark.sql.types import ArrayType, DoubleType
from sklearn.metrics import log_loss, average_precision_score
import os 
import time 
import argparse
import yaml
import numpy as np 
import findspark
findspark.init()

very_start = time.time()

xgbParams = dict(
  eta=0.1,
  maxDepth=8,
  evalMetric='logloss',
  objective='binary:logistic',
  treeMethod='hist',
  missing=0.0,
  seed=42,
  subsample=0.8,
  colsampleBytree=0.8,
  nthread=8,
  numWorkers=16,
  numEarlyStoppingRounds=25,
  numRound=250,
  maximizeEvaluationMetrics=False
)


label_names = ['reply_timestamp', 'retweet_timestamp', 'retweet_with_comment_timestamp', 'like_timestamp']

feature_list = []
feature_list.append(stage2_reply_features[:-1])
feature_list.append(stage2_retweet_features[:-1])
feature_list.append(stage2_comment_features[:-1])
feature_list.append(stage2_like_features[:-1])

# only if you want to train with fixed numbers of trees
num_rounds = [250, 168, 219, 18]

def compute_AP(pred, gt):
    return average_precision_score(gt, pred)

def calculate_ctr(gt):
    positive = len([x for x in gt if x == 1])
    ctr = positive/float(len(gt))
    return ctr

def compute_rce_fast(pred, gt):
    cross_entropy = log_loss(gt, pred)
    yt = np.mean(gt)     
    strawman_cross_entropy = -(yt*np.log(yt) + (1 - yt)*np.log(1 - yt))
    return (1.0 - cross_entropy/strawman_cross_entropy)*100.0

def to_array(col):
    def to_array_(v):
        return v.toArray().tolist()
    return udf(to_array_, ArrayType(DoubleType())).asNondeterministic()(col)

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

    train = spark.read.parquet(data_path+'/stage2_train_pred.parquet')
    print(train.count(), len(train.columns))

    valid = spark.read.parquet(data_path+'/stage2_valid_pred.parquet')
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

    is_local = config['env']['is_local']
    model_save_path = config['files']['model_save_path']
    master = config['env']['master']
    xgb4j_spark_jar = config['training']['xgb4j_spark_jar']
    xgb4j_jar = config['training']['xgb4j_jar']
    hdfs_data_path = config['files']['hdfs_data_path']
    hdfs_model_path = config['files']['hdfs_model_path']
    hdfs_pred_path = config['files']['hdfs_pred_path']
    path_prefix = f"hdfs://{master}:9000/"

    os.environ['PYSPARK_SUBMIT_ARGS'] = f'--jars {xgb4j_spark_jar},{xgb4j_jar} pyspark-shell'

    if is_local:
        print("setting up spark local mode...")
        spark = setup_local()
    else:
        print("setting up spark standalone mode...")
        spark = setup_standalone(master)
    
    train_data, valid_data = read_data(path_prefix+hdfs_data_path)

    sumap = 0
    sumrce = 0

    for i in range(4):

        name = label_names[i]
        print('#'*25);print('###',name);print('#'*25)
        
        vector_assembler = VectorAssembler()\
                            .setInputCols(feature_list[i])\
                            .setOutputCol("features")

        train_data_trans = vector_assembler.setHandleInvalid("keep").transform(train_data)
        valid_data_trans = vector_assembler.setHandleInvalid("keep").transform(valid_data)

        start = time.time()
        xgb_classifier = XGBoostClassifier(**xgbParams).setLabelCol(name)
        #xgb_classifier = XGBoostClassifier(**xgbParams, numRound=num_rounds[i]).setLabelCol(label_names[i])
        xgb_clf_model = xgb_classifier.fit(train_data_trans)
        print(f'Train {name} model took {time.time()-start} seconds')

        start = time.time()
        predictions = xgb_clf_model.transform(valid_data_trans)
        print(f'Predict {name} model took {time.time()-start} seconds')
        
        results = predictions.withColumn("prob", to_array(col("probability")))\
                                    .select(col(name), col("prob")[1])\
                                    .withColumnRenamed("prob[1]", f"pred_{name}").toPandas()
                                          
        gt = results[name]
        pred = results[f"pred_{name}"]
        
        ap = compute_AP(pred, gt)
        rce = compute_rce_fast(pred, gt)
        
        sumap += ap
        sumrce += rce
        txt = f"{name:20} AP:{ap:.5f} RCE:{rce:.5f}"
        print(txt)
     
        print("saving models...")
        xgb_clf_model.write().overwrite().save(path_prefix+f"{hdfs_model_path}/xgboost_{name}_stage2.model")
    
    print('-----------------------------------')
    print("AVG AP: ", sumap/4.)
    print("AVG RCE: ", sumrce/4.)
    
    print('This notebook took %.1f seconds'%(time.time()-very_start))