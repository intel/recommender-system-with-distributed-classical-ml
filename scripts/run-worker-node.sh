dataPath="$1"
tmpPath="$2"
workerNum="$3"
workerIP="$4"
configPath="$5"

workerName="hadoop-slave$workerNum"
repoName=$(basename $(pwd))
repoPath=$(pwd)
currHour=$(date +%Y-%m-%d-%H)

RAM=$(awk '/^Mem/ {print $2}' <(free -mh))
RAM=${RAM/Gi/}
triRAM=$((RAM / 3))
dockerRAM=$((RAM - triRAM))
shmSize=$dockerRAM'g'

echo -e "\nssh to worker ${workerIP}"
ssh ${workerIP} /bin/bash << EOF
cd ${repoPath}

echo -e "\nstart ${workerName} on worker..."
docker run -itd \
                --network host \
                --name ${workerName} \
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
exit
EOF












