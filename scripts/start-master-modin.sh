#!/bin/bash


# helpFunction()
# {
#    echo ""
#    echo "Usage: $0 -d parameterD"
#    echo -e "\t-d path to the data that needed to be mounted into the container"
#    exit 1 # Exit script after printing help
# }

# while getopts "d:" opt
# do
#    case "$opt" in
#       d ) parameterD="$OPTARG" ;;
#       ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
#    esac
# done

# # Print helpFunction in case parameters are empty
# if [ -z "$parameterD" ] 
# then
#    echo "Some or all of the parameters are empty";
#    helpFunction
# fi

# # start hadoop master container
# docker rm -f modin-master &> /dev/null
# echo "start modin-master container..."
# docker run -itd \
#                 --network ray \
#                 --ip 192.168.1.101 \
#                 -p 8265:8265 \
#                 --cap-add=NET_ADMIN \
#                 --name modin-master \
#                 -v $(pwd):/mnt/code \
#                 -v ${parameterD}:/mnt/data \
#                 --shm-size=359gb  \
#                 -e https_proxy=${https_proxy} \
#                 -e http_proxy=${http_proxy} \
#                 -e HTTPS_PROXY=${HTTPS_PROXY} \
#                 -e HTTP_PROXY=${HTTP_PROXY} \
#                 recsys2021:modin 
# sleep 5
# docker exec -d modin-master /bin/bash -c "ip link del dev eth1"
# docker exec -d modin-master /bin/bash -c "ray start --node-ip-address=192.168.1.101 --head --dashboard-host='0.0.0.0' --dashboard-port=8265"
# # get into hadoop master container
# docker exec -it modin-master bash



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

# start hadoop master container
docker rm -f modin-master &> /dev/null
echo "start modin-master container..."
docker run -itd \
                --network host \
                --name modin-master \
                -v $(pwd):/mnt/code \
                -v ${parameterD}:/mnt/data \
                --shm-size=359gb  \
                -e https_proxy=${https_proxy} \
                -e http_proxy=${http_proxy} \
                -e HTTPS_PROXY=${HTTPS_PROXY} \
                -e HTTP_PROXY=${HTTP_PROXY} \
                recsys2021:modin 
sleep 5
# docker exec -d modin-master /bin/bash -c "ip link del dev eth1"
#docker exec -d modin-master /bin/bash -c "ray start --node-ip-address=192.168.1.101 --head --dashboard-host='0.0.0.0' --dashboard-port=8265"
# get into hadoop master container
docker exec -it modin-master bash
