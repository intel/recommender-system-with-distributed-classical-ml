echo -e "\ngenerate workflow tmp folders...."
mkdir -p $1/hdfs/dn 
mkdir -p $1/hdfs/nn
mkdir -p $1/hdfs/tmp 
mkdir -p $1/spark
mkdir -p $1/ray
mkdir -p $1/models
mkdir -p $1/result
mkdir -p $1/logs

chmod -R 777 $1
