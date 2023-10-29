#!/bin/bash

protos=(
    "image_harmony"
    "service_coordinator"
)

for proto in "${protos[@]}"
do
    mkdir -p ./generated/protos/$proto/
    cp ./resources/protos/$proto.proto ./generated/protos/$proto/
    python3 -m grpc_tools.protoc -I . --python_out=. --grpc_python_out=. ./generated/protos/$proto/$proto.proto
    rm ./generated/protos/$proto/$proto.proto
done
