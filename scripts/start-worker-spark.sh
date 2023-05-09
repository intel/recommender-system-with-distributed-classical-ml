dataPath="$1"
hadoopTmpPath="$2"


echo "start hadoop-slave1 container..."
docker rm -f hadoop-slave1 &> /dev/null
docker run -itd \
                --network host \
                --name hadoop-slave1 \
                -v $(pwd):/mnt/code \
                -v ${dataPath}:/mnt/data \
                -v ${hadoopTmpPath}/tmp:/mnt/tmp \
                -e HTTPS_PROXY=$HTTPS_PROXY \
                -e HTTP_PROXY=$HTTP_PROXY \
		-e https_proxy=$https_proxy \
                -e http_proxy=$http_proxy \
                recsys2021:spark &> /dev/null


# get into hadoop slave container
docker exec -it hadoop-slave1 bash
