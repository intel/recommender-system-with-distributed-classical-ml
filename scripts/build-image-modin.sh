#!/bin/bash

echo ""
echo -e "\nbuild docker hadoop image\n"
docker build -f Dockerfile-modin -t recsys2021:modin \
            --build-arg https_proxy=${https_proxy} \
            --build-arg http_proxy=${http_proxy} .
echo ""
