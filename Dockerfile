FROM ubuntu:22.04 as base

# See http://bugs.python.org/issue19846
ENV LANG C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y --no-install-recommends --fix-missing \
    ca-certificates \
    curl \
    python3 \
    python3-distutils \
    python3-dev \
    build-essential

RUN curl -fSsL https://bootstrap.pypa.io/get-pip.py | python3

RUN ln -sf $(which python3) /usr/local/bin/python && \
    ln -sf $(which python3) /usr/local/bin/python3 && \
    ln -sf $(which python3) /usr/bin/python

RUN python -m pip --no-cache-dir install --upgrade \
    pip \
    setuptools

RUN apt-get -y update && \
    apt-get -y install --no-install-recommends openssh-server wget vim net-tools git-all htop && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# install java
RUN wget --no-check-certificate -q https://repo.huaweicloud.com/java/jdk/8u201-b09/jdk-8u201-linux-x64.tar.gz && \
    tar -zxvf jdk-8u201-linux-x64.tar.gz && \
    mv jdk1.8.0_201 /opt/jdk1.8.0_201 && \
    rm jdk-8u201-linux-x64.tar.gz
 
# install hadoop 3.3.3
RUN wget --no-check-certificate -q https://dlcdn.apache.org/hadoop/common/hadoop-3.3.3/hadoop-3.3.3.tar.gz && \
    tar -zxvf hadoop-3.3.3.tar.gz && \
    mv hadoop-3.3.3 /opt/hadoop-3.3.3 && \
    rm hadoop-3.3.3.tar.gz
 
# install spark 3.3.1
RUN wget --no-check-certificate https://archive.apache.org/dist/spark/spark-3.3.1/spark-3.3.1-bin-hadoop3.tgz && \
    tar -zxvf spark-3.3.1-bin-hadoop3.tgz && \
    mv spark-3.3.1-bin-hadoop3 /opt/spark-3.3.1-bin-hadoop3 && \
    rm spark-3.3.1-bin-hadoop3.tgz
 
ENV HADOOP_HOME=/opt/hadoop-3.3.3
ENV JAVA_HOME=/opt/jdk1.8.0_201
ENV JRE_HOME=$JAVA_HOME/jre
ENV SPARK_HOME=/opt/spark-3.3.1-bin-hadoop3
ENV PYSPARK_PYTHON=/usr/local/bin/python
ENV PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$SPARK_HOME/bin
ENV PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9.5-src.zip:$PYTHONPATH
ENV HADOOP_COMMON_LIB_NATIVE_DIR=$HADOOP_HOME/lib/native
ENV HADOOP_OPTS="$HADOOP_OPTS -Djava.library.path=$HADOOP_HOME/lib/native"
ENV LD_LIBRARY_PATH=$HADOOP_HOME/lib/native
 
RUN ssh-keygen -t rsa -f /root/.ssh/id_rsa -P '' && \
    cat /root/.ssh/id_rsa.pub >> /root/.ssh/authorized_keys && \
    sed -i 's/#   Port 22/Port 12345/' /etc/ssh/ssh_config && \
    sed -i 's/#Port 22/Port 12345/' /etc/ssh/sshd_config

RUN pip install --no-cache-dir pyarrow findspark numpy pandas transformers pyrecdp scikit-learn category_encoders ray[tune]==2.2.0 xgboost xgboost-ray optuna sigopt pyyaml raydp daal4py simplejson modin[ray]
RUN pip install git+https://github.com/sllynn/spark-xgboost.git

WORKDIR /workspace

RUN wget -P /workspace/third_party https://repo1.maven.org/maven2/ml/dmlc/xgboost4j-spark_2.12/1.5.2/xgboost4j-spark_2.12-1.5.2.jar && \
    wget -P /workspace/third_party https://repo1.maven.org/maven2/ml/dmlc/xgboost4j_2.12/1.5.2/xgboost4j_2.12-1.5.2.jar

CMD ["sh", "-c", "service ssh start; bash"]