dataPath="$1"
tmpPath="$2"
yamlPath="$3"
configPath="$4"

repoName=$(basename $(pwd))

currHour=$(date +%Y-%m-%d-%H)

RAM=$(awk '/^Mem/ {print $2}' <(free -mh))
RAM=${RAM/Gi/}
triRAM=$((RAM / 3))
dockerRAM=$((RAM - triRAM))
shmSize=$dockerRAM'g'

# start hadoop master container
echo -e "\nstart hadoop-master container...\n"
docker run \
        --network host \
        --name hadoop-master \
        --shm-size ${shmSize} \
        -v $(pwd):/workspace/${repoName} \
        -v ${configPath}:/workspace/configs \
        -v ${dataPath}:/workspace/data \
        -v ${tmpPath}:/workspace/tmp \
        -w /workspace/${repoName} \
        -e HTTPS_PROXY=$HTTPS_PROXY \
        -e HTTP_PROXY=$HTTP_PROXY \
        -e https_proxy=$https_proxy \
        -e http_proxy=$http_proxy \
        classical-ml-wf:latest python start-workflow.py --config-file ${yamlPath}

echo -e "\nshut down cluster...\n"
docker rm -f hadoop-master 