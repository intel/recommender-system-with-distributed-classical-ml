echo -e "\ncreate recsys2021 HDFS data folder"
  
hdfs dfs -mkdir -p /recsys2021/stage1
hdfs dfs -mkdir -p /recsys2021/stage2
hdfs dfs -mkdir -p /recsys2021/models

echo -e "\nload data into HDFS folder"
hdfs dfs -put /mnt/data/processed4/train/stage1/train /recsys2021/stage1
hdfs dfs -put /mnt/data/processed4/train/stage1/valid /recsys2021/stage1
hdfs dfs -put /mnt/data/processed4/train/stage2/train /recsys2021/stage2
hdfs dfs -put /mnt/data/processed4/train/stage2/valid /recsys2021/stage2

echo -e "\ncheck whether data is there"
hdfs dfs -ls /recsys2021/stage1
hdfs dfs -ls /recsys2021/stage2