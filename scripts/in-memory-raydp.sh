!/usr/bin/env bash

echo "processing training data..."
SECONDS=0
python src/data_preprocess/spark/datapre.py train mlp-prod-icx-108398
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "processing validation data for stage 1..."
SECONDS=0
python src/data_preprocess/spark/datapre.py valid_stage1 mlp-prod-icx-108398
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "processing validation data for stage 2..."
SECONDS=0
python src/data_preprocess/spark/datapre.py valid_stage2 mlp-prod-icx-108398
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "downloading data for training..."
SECONDS=0
# make a directory to store processed data
mkdir /mnt/data/processed_data 
# download pre-prcocessed training dataset
hdfs dfs -copyToLocal /recsys2021/datapre_stage1/stage1_train /mnt/data/processed_data
hdfs dfs -copyToLocal /recsys2021/datapre_stage1/stage1_valid /mnt/data/processed_data
# download preprocessed validation dataset
hdfs dfs -copyToLocal /recsys2021/datapre_stage2/stage2_train /mnt/data/processed_data
hdfs dfs -copyToLocal /recsys2021/datapre_stage2/stage2_valid /mnt/data/processed_data
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "copying data for worker..."
SECONDS=0
scp -r /mnt/data/processed_data mlp-prod-icx-108397:/mnt/data/
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

echo "stage-1 training..."
SECONDS=0
python src/train_models/xgboost/ray/train_stage1_ray.py --config-dir /mnt/code/config.yaml
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

mkdir /mnt/data/processed_data/stage2_train2
mkdir /mnt/data/processed_data/stage2_valid2

echo "data merging..."
SECONDS=0
python src/train_models/xgboost/ray/train_merge12_ray.py --config-dir /mnt/code/config.yaml
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


echo "copying data for worker..."
SECONDS=0
scp -r /mnt/data/processed_data/stage2_train2 mlp-prod-icx-108397:/mnt/data/processed_data/
scp -r /mnt/data/processed_data/stage2_valid2 mlp-prod-icx-108397:/mnt/data/processed_data/
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


echo "stage-2 training..."
SECONDS=0
python src/train_models/xgboost/ray/train_stage2_ray.py --config-dir /mnt/code/config.yaml
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."









