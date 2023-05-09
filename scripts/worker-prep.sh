
#!/bin/bash

workerIP="$1"
tmpPath="$2"

repoPath=$(pwd)
tarPath="$(dirname "$repoPath")"

mv .git ../
echo -e "\ncopy workflow repo to worker ${workerIP}"
scp -q -r ${repoPath} ${workerIP}:${tarPath}/
mv ../.git ./

echo -e "\nssh to worker ${workerIP}"
ssh ${workerIP} /bin/bash << EOF
cd ${repoPath}
bash ./scripts/tmp-folder-prep.sh ${tmpPath}
exit
EOF

