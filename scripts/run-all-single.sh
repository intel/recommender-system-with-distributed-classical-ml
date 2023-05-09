#!/bin/bash
echo -e "\nprepare spark dev environment...."
./scripts/prep-env.sh -a hadoop-master -b hadoop-master -c 0 -d 0
echo -e "\n"
sleep 10

echo -e "\nload data into hdfs...."
./scripts/recsys-data-prep.sh
echo -e "\n"

echo -e "\nstart data preprocessing...."
cd /mnt/code/src/data_preprocess/spark
python datapre.py train hadoop-master
python datapre.py valid_stage1 hadoop-master
python datapre.py valid_stage2 hadoop-master
echo -e "\n"
sleep 10
echo -e "\ndownload data to local...."
mkdir /mnt/data/processed_data
hdfs dfs -copyToLocal /recsys2021/datapre_stage1/stage1_train /mnt/data/processed_data
hdfs dfs -copyToLocal /recsys2021/datapre_stage1/stage1_valid /mnt/data/processed_data
hdfs dfs -copyToLocal /recsys2021/datapre_stage2/stage2_train /mnt/data/processed_data
hdfs dfs -copyToLocal /recsys2021/datapre_stage2/stage2_valid /mnt/data/processed_data
echo -e "\n"
sleep 10
echo -e "\nstart training stage 1...."
python /mnt/code/src/train_models/xgboost/train_stage1.py --config-dir /mnt/code/config.yaml
echo -e "\n"

echo -e "\ncreate data for stage 2...."
python /mnt/code/src/train_models/xgboost/train_merge12.py --config-dir /mnt/code/config.yaml
echo -e "\n"

echo -e "\nstart training stage 2...."
python /mnt/code/src/train_models/xgboost/train_stage2.py --config-dir /mnt/code/config.yaml
echo -e "\n"

echo -e "\nall training finished!"
