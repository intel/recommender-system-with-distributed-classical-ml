#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -d parameterD -h parameterH -p parameterP -s parameterS"
   echo -e "\t-d path to the data that needed to be mounted into the container"
   echo -e "\t-h path to the hadoop tmp dir that needed to be mounted into the container"
   echo -e "\t-p your http proxy address, you can find out by typing env in linux command line"
   echo -e "\t-s your https proxy address, you can find out by typing env in linux command line"
   exit 1 # Exit script after printing help
}

while getopts "d:h:p:s:" opt
do
   case "$opt" in
      d ) parameterD="$OPTARG" ;;
      h ) parameterH="$OPTARG" ;;
      p ) parameterP="$OPTARG" ;;
      s ) parameterS="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterD" ] || [ -z "$parameterH" ]|| [ -z "$parameterP" ]|| [ -z "$parameterS" ]
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


echo "start hadoop-slave1 container..."
docker rm -f hadoop-slave1 &> /dev/null
docker run -itd \
                --network ray \
                --cap-add=NET_ADMIN \
                --name hadoop-slave1 \
                --shm-size=250gb \
                -v $(pwd):/mnt/code \
                -v ${parameterH}:/mnt/tmp \
		-v ${parameterD}:/mnt/data \
                -e https_proxy=${parameterS} \
                -e http_proxy=${parameterP} \
                -e HTTPS_PROXY=${parameterS} \
                -e HTTP_PROXY=${parameterP} \
                recsys2021:3.0 &> /dev/null
sleep 5
docker exec -d hadoop-slave1 /bin/bash -c "ip link del dev eth1"
docker exec -d hadoop-slave1 /bin/bash -c "ray start --address=192.168.1.101:6379"
# get into hadoop slave container
docker exec -it hadoop-slave1 bash
