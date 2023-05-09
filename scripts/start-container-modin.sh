#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -d parameterD"
   echo -e "\t-d path to the data that needed to be mounted into the container"
   exit 1 # Exit script after printing help
}

while getopts "d:" opt
do
   case "$opt" in
      d ) parameterD="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$parameterD" ] 
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


docker rm -f modin-master &> /dev/null
echo "start modin-master container..."
docker run -itd \
                --net=hadoop \
                -p 8088:8088 \
                -p 8888:8888 \
                -p 8080:8080 \
                -p 9870:9870 \
                -p 9864:9864 \
                -p 4040:4040 \
                -p 18081:18081 \
                -p 10086:22 \
                --shm-size 150g \
                --name modin-master \
                --hostname modin-master \
                -v $(pwd):/mnt/code \
                -v ${parameterD}:/mnt/data \
                -e https_proxy=${https_proxy} \
                -e http_proxy=${http_proxy} \
                -e HTTPS_PROXY=${HTTPS_PROXY} \
                -e HTTP_PROXY=${HTTP_PROXY} \
                recsys2021:modin &> /dev/null

docker exec -it modin-master bash
