#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -a parameterA -b parameterB -c parameterC -d parameterD"
   echo -e "\t-a hostname of the master node"
   echo -e "\t-b hostname of the worker node, if single-node, this should be the same with master node"
   echo -e "\t-c whether it is a master node, 0 stands for master, 1 stands for slave"
   echo -e "\t-d cluster mode, 0 stands for single-node and single-containers, 1 stands for multi-node"
   exit 1 # Exit script after printing help
}

while getopts "a:b:c:d:" opt
do
   case "$opt" in
      a ) parameterA="$OPTARG" ;;
      b ) parameterB="$OPTARG" ;;
      c ) parameterC="$OPTARG" ;;
      d ) parameterD="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterA" ] || [ -z "$parameterB" ] || [ -z "$parameterC" ] || [ -z "$parameterD" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

MAIN=$parameterA
SECONDARY=$parameterB
SPARK_VERSION=`echo $SPARK_HOME | cut -d "-" -f 2`
HOME_DIR=$(pwd)


cp $HOME_DIR/configs/hadoop-env.sh $HADOOP_HOME/etc/hadoop/hadoop-env.sh && \
cp $HOME_DIR/configs/hdfs-site.xml $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
cp $HOME_DIR/configs/core-site.xml $HADOOP_HOME/etc/hadoop/core-site.xml && \
cp $HOME_DIR/configs/mapred-site.xml $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
cp $HOME_DIR/configs/yarn-site.xml $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
cp $HOME_DIR/configs/slaves $HADOOP_HOME/etc/hadoop/slaves && \
cp $HOME_DIR/configs/ssh_config /root/.ssh/config && \
cp $HOME_DIR/configs/spark-env.sh $SPARK_HOME/conf/spark-env.sh && \
cp $HOME_DIR/configs/spark-defaults.conf $SPARK_HOME/conf/spark-defaults.conf 

sed -i 's@hadoop-master@'"$MAIN"'@'  $HADOOP_HOME/etc/hadoop/core-site.xml && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $HADOOP_HOME/etc/hadoop/hdfs-site.xml && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $HADOOP_HOME/etc/hadoop/yarn-site.xml && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $HADOOP_HOME/etc/hadoop/mapred-site.xml && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $SPARK_HOME/conf/spark-defaults.conf && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $SPARK_HOME/conf/spark-env.sh && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $HADOOP_HOME/etc/hadoop/slaves && \
sed -i 's@hadoop-master@'"$MAIN"'@'  $HADOOP_HOME/etc/hadoop/workers && \
sed -i 's@hadoop-slave1@'"$SECONDARY"'@' $HADOOP_HOME/etc/hadoop/slaves && \
sed -i 's@hadoop-slave1@'"$SECONDARY"'@' $HADOOP_HOME/etc/hadoop/workers && \


if [[ $parameterC = "0" ]]
then
    echo -e "\nformat namenode..."
    (sleep 10; echo y) | $HADOOP_HOME/bin/hdfs namenode -format
else 
    echo -e "\nformat datanode..."
    (sleep 10; echo y) | $HADOOP_HOME/bin/hdfs datanode -format
fi


if [[ $parameterD = "0" ]]; then 
    echo -e "\nSetting Up Single-Node&Single-Container Version of the Spark Cluster..."   
    echo -e "\nstart HDFS..."
    $HADOOP_HOME/sbin/start-dfs.sh
    echo -e "\n"

    if [[ $parameterC = "0" ]]
    then
        echo -e "\nstart spark master..."
        source $SPARK_HOME/conf/spark-env.sh 
        $SPARK_HOME/sbin/start-master.sh
        
        echo -e "\nstart spark worker..."
        $SPARK_HOME/sbin/start-worker.sh spark://$MAIN:7077 

        echo -e "\ncreate spark history folder..."
        hdfs dfs -mkdir -p /spark/history

        echo -e "\nstart spark history..."
        $SPARK_HOME/sbin/start-history-server.sh
    else 
        echo -e "\nstart spark worker..."
        source $SPARK_HOME/conf/spark-env.sh 
        $SPARK_HOME/sbin/start-worker.sh spark://$MAIN:7077 
    fi

    echo -e "\n"
    
elif [[ $parameterD = "1" ]]; then 
    echo "Setting Up 2-Node&2-Container Version of the Spark Cluster..."

    echo "Starting Hadoop..."

    $HADOOP_HOME/sbin/start-dfs.sh

    if [[ $parameterC = "0" ]]
    then
        echo -e "\nstart spark..."
        source $SPARK_HOME/conf/spark-env.sh 
        $SPARK_HOME/sbin/start-master.sh
        echo -e "\ncreate spark history folder..."
        hdfs dfs -mkdir -p /spark/history

        if [[ $SPARK_VERSION == '3.3.0' ]]
        then 
            echo -e "\nstart spark worker..."
            $SPARK_HOME/sbin/start-worker.sh spark://$MAIN:7077
        else 
            echo -e "\nstart spark worker..."
            $SPARK_HOME/sbin/start-slave.sh spark://$MAIN:7077
        fi 
    else 
        echo -e "\nstart spark master..."
        source $SPARK_HOME/conf/spark-env.sh 
        
        if [[ $SPARK_VERSION == '3.3.0' ]]
        then 
            echo -e "\nstart spark worker..."
            $SPARK_HOME/sbin/start-worker.sh spark://$MAIN:7077
        else
            echo -e "\nstart spark worker..."
            $SPARK_HOME/sbin/start-slave.sh spark://$MAIN:7077
        fi
    fi
fi 


echo -e "\ntest with the Pi Example..."
$SPARK_HOME/bin/spark-submit --class org.apache.spark.examples.SparkPi --master spark://$MAIN:7077 $SPARK_HOME/examples/jars/spark-examples_2.12-$SPARK_VERSION.jar 10







