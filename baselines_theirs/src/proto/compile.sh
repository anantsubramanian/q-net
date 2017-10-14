#!/bin/bash
# Compiles the protocol buffer files. To use, please download and install
# the protocol buffer compiler from
# https://developers.google.com/protocol-buffers/docs/downloads

DIR=$(dirname $0)

PROTOS=$(cd $DIR && ls *.proto | sed 's/\.proto//g')

for proto in ${PROTOS[@]}; do
	protoc -I=$DIR --python_out=$DIR --cpp_out=$DIR $DIR/$proto.proto
done
