dataPath="$1"
tmpPath="$2"
configPath="$3"
repoName=$(basename $(pwd))

currHour=$(date +%Y-%m-%d-%H)
# start hadoop master container

RAM=$(awk '/^Mem/ {print $2}' <(free -mh))
RAM=${RAM/Gi/}
triRAM=$((RAM / 3))
dockerRAM=$((RAM - triRAM))
shmSize=$dockerRAM'g'

echo -e "\nstart hadoop-master on master..."
docker run -itd \
                --network host \
                --name hadoop-master \
                -v $(pwd):/workspace/${repoName} \
                -v ${configPath}:/workspace/configs \
                -v ${dataPath}:/workspace/data \
                -v ${tmpPath}:/workspace/tmp \
                -w /workspace/${repoName} \
                --shm-size ${shmSize}  \
                -e HTTPS_PROXY=$HTTPS_PROXY \
                -e HTTP_PROXY=$HTTP_PROXY \
                -e https_proxy=$https_proxy \
                -e http_proxy=$http_proxy \
                classical-ml-wf:latest |& tee -a ${tmpPath}/logs/launch-${currHour}.log

