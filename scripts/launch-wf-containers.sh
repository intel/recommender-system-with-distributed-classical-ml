#!/bin/bash

data_path="$1"
tmp_path="$2"
config_path="$3"
wf_abs_path="$4"
model_path="$5"
worker_name="$6"


repo_name=$(basename $wf_abs_path)

curr_hour=$(date +%Y-%m-%d-%H)

RAM=$(awk '/^Mem/ {print $2}' <(free -mh))
RAM=${RAM/Gi/}
tri_RAM=$((RAM / 3))
docker_ram=$((RAM - tri_RAM))
shm_size=$docker_ram'g'


echo -e "\nlaunch workflow containers..."
docker run -itd \
        --network host \
        --name ${worker_name} \
        -v ${wf_abs_path}:/workspace/${repo_name} \
        -v ${config_path}:/workspace/configs \
        -v ${data_path}:/workspace/data \
        -v ${tmp_path}:/workspace/tmp \
        -v ${model_path}:/MODELS \
        -w /workspace/${repo_name} \
        --shm-size ${shm_size}  \
        -e HTTPS_PROXY=$HTTPS_PROXY \
        -e HTTP_PROXY=$HTTP_PROXY \
        -e https_proxy=$https_proxy \
        -e http_proxy=$http_proxy \
        -e training_task=${TRAINING_TASK:-credit_card} \
        intel/ai-workflows:pa-fraud-detection-classical-ml |& tee -a ${tmp_path}/logs/launch-${curr_hour}.log

