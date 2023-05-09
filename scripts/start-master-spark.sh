dataPath="$1"
hadoopTmpPath="$2"

# start hadoop master container
docker rm -f hadoop-master &> /dev/null
echo "start hadoop-master container..."
docker run -itd \
                --network host \
                --name hadoop-master \
                -v $(pwd):/mnt/code \
                -v ${dataPath}:/mnt/data \
                -v ${hadoopTmpPath}/tmp:/mnt/tmp \
                -e HTTPS_PROXY=$HTTPS_PROXY \
                -e HTTP_PROXY=$HTTP_PROXY \
                -e https_proxy=$https_proxy \
                -e http_proxy=$http_proxy \
                recsys2021:spark

# get into hadoop master container
docker exec -it hadoop-master bash


