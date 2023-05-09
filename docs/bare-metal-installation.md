# Hadoop&Spark Bare Metal Installation Guide  

The following guide shows the detailed steps for 2-node version of the env setup. For multi-node beyond 2 nodes, the steps are very similar. We assume that you have `mlp-opa-clx-4021` as your master node and `mlp-opa-clx-4025` as your worker node.

## 1. create folders on master node
```bash
# assume you are in the folder `/home`
mkdir big-data-env
mkdir -p hdfs/tmp
mkdir -p hdfs/nn
mkdir -p hdfs/dn
cd big-data-env
```

## 2. download the binary files of Spark, Hadoop, Java 
```bash
# download
wget --no-check-certificate https://dlcdn.apache.org/hadoop/common/hadoop-3.2.3/hadoop-3.2.3.tar.gz
wget --no-check-certificate https://dlcdn.apache.org/spark/spark-3.1.3/spark-3.1.3-bin-hadoop3.2.tgz
wget --no-check-certificate https://repo.huaweicloud.com/java/jdk/8u201-b09/jdk-8u201-linux-x64.tar.gz

# unpack 
tar -zxvf hadoop-3.2.3.tar.gz
tar -zxvf spark-3.1.3-bin-hadoop3.2.tgz
tar -zxvf jdk-8u201-linux-x64.tar.gz
```

## 3. configure Hadoop config files and move source code to lib folder 
```bash
cd hadoop-3.2.3

# edit the hadoop-env.sh file
vim etc/hadoop/hadoop-env.sh

# add the following to the end of this file and save
export JAVA_HOME=/opt/jdk1.8.0_201
export HDFS_NAMENODE_USER=root
export HDFS_DATANODE_USER=root
export HDFS_SECONDARYNAMENODE_USER=root
export YARN_RESOURCEMANAGER_USER=root
export YARN_NODEMANAGER_USER=root

# edit the slave file
vim etc/hadoop/slaves

# add the following and save
mlp-opa-clx-4021
mlp-opa-clx-4025

# edit the core-site.xml file
vim etc/hadoop/core-site.xml

# add the following to the configuration brackets and save
    <property>
            <name>hadoop.tmp.dir</name>
            <value>/home/hdfs/tmp</value>
        </property>
        <property>
            <name>fs.defaultFS</name>
            <value>hdfs://mlp-opa-clx-4025:9000</value>
    </property>


# edit the hdfs-site.xml file
vim etc/hadoop/hdfs-site.xml

#  add the following to the configuration brackets and save 
         <property>
            <name>dfs.replication</name>
            <value>1</value>
        </property>
     
        <property>
        <name>dfs.namenode.name.dir</name>
        <value>/home/hdfs/nn</value>
      </property>
     
       <property>
        <name>dfs.datanode.data.dir</name>
        <value>/home/hdfs/dn</value>
      </property>
     
        <property>
            <name>dfs.namenode.http-address</name>
            <value>mlp-opa-clx-4025:9870</value>
        </property>
        <property>
            <name>dfs.namenode.secondary.http-address</name>
            <value>mlp-opa-clx-4025:9868</value>
    </property>

# edit the mapred-site.xml file
vim etc/hadoop/mapred-site.xml

#  add the following to the configuration brackets and save 
  <property>
            <name>mapreduce.framework.name</name>
            <value>yarn</value>
       </property>
       <property>
            <name>mapreduce.jobhistory.address</name>
            <value>mlp-opa-clx-4025:10020</value>
       </property>
       <property>
            <name>mapreduce.jobhistory.webapp.address</name>
            <value>mlp-opa-clx-4025:19888</value>
       </property>
 

# mv all the libs to lib folder
cd ..
mkdir lib
mv spark-3.1.3-bin-hadoop3.2 lib
mv scala-2.12.12 lib
mv hadoop-3.2.3 lib
mv jdk1.8.0_201 lib
rm jdk-8u201-linux-x64.tar.gz hadoop-3.2.3.tar.gz spark-3.1.3-bin-hadoop3.2.tgz scala-2.12.12.tgz
```

## 4. password-less ssh 
```bash
ssh-keygen -t rsa -f ~/.ssh/id_rsa -P ''
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys
scp ~/.ssh/authorized_keys mlp-opa-clx-4021:~/.ssh/authorized_keys
## try ssh to test whether it works with password-less ssh
```

## 5. copy files over to worker node 
```bash
scp -p /home/big-data-env mlp-opa-clx-4021:/home/big-data-env

```

## 6. set the Hadoop&Spark env variables in ./bashrc
```bash
vim ~/.bashrc
 
# append the following and save
JAVA_HOME=/home/big-data-env/jdk1.8.0_201
JAVA_BIN=${JAVA_HOME}/bin
JRE_HOME=${JAVA_HOME}/jre
export  JAVA_HOME  JAVA_BIN JRE_HOME 
export SCALA_HOME=/home/big-data-env/scala-2.12.12
export HADOOP_HOME=/home/big-data-env/hadoop-3.2.3
export PATH=$PATH:$JAVA_BIN:${SCALA_HOME}/bin:${HADOOP_HOME}/bin:${HADOOP_HOME}/sbin
export JAVA_HOME=/home/big-data-env/jdk1.8.0_201
export PATH=$JAVA_HOME/bin:$PATH
export HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
export HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
export LD_LIBRARY_PATH=$HADOOP_HOME/lib/native
 
source ~/.bashrc
# check whether it works
java -version
hadoop version
```

## 7. start hadoop/spark cluster 
```bash
## on master node 
/home/big-data-env/hadoop-3.2.3/bin/hdfs namenode -format

# start hdfs
/home/big-data-env/hadoop-3.2.3/sbin/start-dfs.sh

# start spark
/home/big-data-env/spark-3.1.3-bin-hadoop3.2/sbin/start-master.sh

## on worker node 

# start hdfs
/home/big-data-env/hadoop-3.2.3/sbin/start-dfs.sh

# start spark
/opt/spark-3.1.3-bin-hadoop3.2/sbin/start-worker.sh spark://hadoop0:7077
```


## 8. test whether it works with the spark cluster
```bash
# on master node 

/home/big-data-env/spark-3.1.3-bin-hadoop3.2/bin/spark-submit --class org.apache.spark.examples.SparkPi --master spark://hadoop0:7077 /home/big-data-env/spark-3.1.3-bin-hadoop3.2/examples/jars/spark-examples_2.12-3.1.3.jar 10

``` 