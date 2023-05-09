#!/bin/bash

echo ""
echo -e "\nbuild docker hadoop image\n"
docker build -f Dockerfile-spark -t recsys2021:spark \
            --build-arg https_proxy=$https_proxy \
            --build-arg http_proxy=$http_proxy .
echo ""
