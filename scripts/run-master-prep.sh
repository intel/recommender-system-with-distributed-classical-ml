#!/bin/bash

masterIP="$1"
dataPath="$2"
tmpPath="$3"
yamlPath="$4"
isMulti="$5"

repoPath=$(pwd)
tarPath="$(dirname "$repoPath")"

mv .git ../
echo -e "\ncopy workflow repo to worker ${masterIP}"
scp -q -r ${repoPath} ${masterIP}:${tarPath}/
mv ../.git ./


echo -e "\nssh to master ${masterIP}"
ssh ${masterIP} /bin/bash << EOF
cd ${repoPath}
bash ./scripts/tmp-folder-prep.sh ${tmpPath}
exit
EOF

if [ $isMulti = "0" ]; then
    ssh ${masterIP} "
        cd ${repoPath} 
        bash ./scripts/run-single-node.sh ${dataPath} ${tmpPath} ${yamlPath}
    "
fi 





