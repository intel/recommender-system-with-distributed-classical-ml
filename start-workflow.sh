#!/bin/bash

yamlPath="$1"

function parse_yaml {
   local prefix=$2
   local s='[[:space:]]*' w='[a-zA-Z0-9_]*' fs=$(echo @|tr @ '\034')
   sed -ne "s|,$s\]$s\$|]|" \
        -e ":1;s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s,$s\(.*\)$s\]|\1\2: [\3]\n\1  - \4|;t1" \
        -e "s|^\($s\)\($w\)$s:$s\[$s\(.*\)$s\]|\1\2:\n\1  - \3|;p" $1 | \
   sed -ne "s|,$s}$s\$|}|" \
        -e ":1;s|^\($s\)-$s{$s\(.*\)$s,$s\($w\)$s:$s\(.*\)$s}|\1- {\2}\n\1  \3: \4|;t1" \
        -e    "s|^\($s\)-$s{$s\(.*\)$s}|\1-\n\1  \2|;p" | \
   sed -ne "s|^\($s\):|\1|" \
        -e "s|^\($s\)-$s[\"']\(.*\)[\"']$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)-$s\(.*\)$s\$|\1$fs$fs\2|p" \
        -e "s|^\($s\)\($w\)$s:$s[\"']\(.*\)[\"']$s\$|\1$fs\2$fs\3|p" \
        -e "s|^\($s\)\($w\)$s:$s\(.*\)$s\$|\1$fs\2$fs\3|p" | \
   awk -F$fs '{
      indent = length($1)/2;
      vname[indent] = $2;
      for (i in vname) {if (i > indent) {delete vname[i]; idx[i]=0}}
      if(length($2)== 0){  vname[indent]= ++idx[indent] };
      if (length($3) > 0) {
         vn=""; for (i=0; i<indent; i++) { vn=(vn)(vname[i])("_")}
         printf("%s%s%s=\"%s\"\n", "'$prefix'",vn, vname[indent], $3);
      }
   }'
}

eval $(parse_yaml $yamlPath)

repoPath=$(pwd)
repoName=$(basename $(pwd))
hostName=$(hostname -I | awk '{print $1}')
currTime=$(date +%Y-%m-%d-%H-%M)
wfTmpFolder="$env_tmp_path/wf-session-${currTime}"


if [ "$data_preprocess_dp_framework" == "spark" ] && [ "$training_train_framework" == "spark" ]; then 
    clusterEngine='spark'
elif [ "$data_preprocess_dp_framework" == 'spark' ] && [ -z "$training_train_framework" ]; then 
    clusterEngine='spark'
elif [ -z "$data_preprocess_dp_framework" ] && [ "$training_train_framework" == 'spark' ]; then 
    clusterEngine='spark'
else
    clusterEngine='ray'
fi 

if [ -z "${env_node_ips_2}" ]; then   
    echo -e "\nsetting up single-node cluster..."

    if [ "$hostName" == "$env_node_ips_1" ]; then
        bash ./scripts/tmp-folder-prep.sh "$wfTmpFolder"
        bash ./scripts/run-single-node.sh "$env_data_path" "$wfTmpFolder" "$yamlPath" "$env_config_path"
    else
        bash ./scripts/run-master-prep.sh "${env_node_ips_1}" "${env_data_path}" "${wfTmpFolder}" "${yamlPath}" 0
    fi
    
else 
    echo -e "\nsetting up multi-node cluster..."
    echo -e "\non master node..."
    if [ "$hostName" == "$env_node_ips_1" ]; then
        bash ./scripts/tmp-folder-prep.sh "$wfTmpFolder"
    else
        bash ./scripts/run-master-prep.sh "${env_node_ips_1}" "${env_data_path}" "${wfTmpFolder}" "${yamlPath}" 1
    fi 

    for ((i=2; i<=$env_num_node; i++)); do
        workerIP="env_node_ips_$i"
        bash ./scripts/worker-prep.sh "${!workerIP}" "$wfTmpFolder"
    done

    if [ "$clusterEngine" == 'ray' ]; then
        echo -e "\nlaunch ray cluster..."
        if [ "$hostName" == "$env_node_ips_1" ]; then
            bash ./scripts/run-master-node.sh "$env_data_path" "$wfTmpFolder" "$env_config_path"
            docker exec hadoop-master ray start --head --port=6379 --temp-dir=/workspace/tmp/ray
        else 
            ssh ${env_node_ips_1} "
                bash ./scripts/run-master-node.sh "$env_data_path" "$wfTmpFolder" 
                docker exec hadoop-master ray start --head --port=6379 --temp-dir=/workspace/tmp/ray
            "
        fi 

        for ((i=2; i<=$env_num_node; i++)); do
            workerIP="env_node_ips_$i"
            workerNum=$((i-1))
            workerName="hadoop-slave$workerNum"
            masterPort="$env_node_ips_1:6379"
            bash ./scripts/run-worker-node.sh "$env_data_path" "$wfTmpFolder" "$workerNum" "${!workerIP}" "$env_config_path"
            ssh ${!workerIP} "
                docker exec $workerName ray start --address=$masterPort --temp-dir=/workspace/tmp/ray
            "
        done

    elif [ "$clusterEngine" == 'spark' ]; then 
        echo "launch spark cluster..."
        echo "To be implemented"
    else
        echo "Unknown Cluster Engine"
    fi 
    

    echo -e "\nstart workflow on master...\n"
    if [ "$hostName" == "$env_node_ips_1" ]; then
        docker exec -i hadoop-master python start-workflow.py --config-file ${yamlPath}
        if [ ! -z "${data_preprocess_input_data_path}" ]; then
            echo "collecting files from nodes to master..."
            docker exec -i hadoop-master python src/utils/collect_modin_files.py ${yamlPath}
        fi
    else 
        ssh ${env_node_ips_1} "
        cd ${repoPath}
        docker exec -i hadoop-master python start-workflow.py --config-file ${yamlPath}
    "
    fi 
    
    echo -e "\nshut down cluster...\n"
    if [ "$hostName" == "$env_node_ips_1" ]; then
        docker rm -f hadoop-master 
    else 
        ssh ${env_node_ips_1} "
            docker rm -f hadoop-master
            rm -fr ${repoPath}
        " 
    fi 

    for ((i=2; i<=$env_num_node; i++)); do
        workerIP="env_node_ips_$i"
        workerNum=$((i-1))
        workerName="hadoop-slave$workerNum"
        ssh ${!workerIP} "
            docker rm -f ${workerName}
            rm -fr ${repoPath}
        "
    done

fi