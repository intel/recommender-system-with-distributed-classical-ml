echo -e "\nremove path if already exists...."
sudo rm -fr $1/tmp

echo -e "\ncreate folder for hadoop...."
sudo mkdir -p $1/tmp/hdfs/dn 
sudo mkdir -p $1/tmp/hdfs/nn
sudo mkdir -p $1/tmp/hdfs/tmp 
sudo mkdir -p $1/tmp/spark
sudo mkdir -p $1/tmp/models
sudo mkdir -p $1/tmp/result