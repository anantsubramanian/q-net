#!/bin/bash

#-------------------------------------REQUIRES:--------------------------------#
#                                                                              #
# 1. TensorFlow v1.2                                                          #
# 2. Training and dev feature files in PROTO_DUMPS_PATH folder                 #
# 3. SQuAD v1.0 train and dev json files                                       #
#                                                                              #
#------------------------------------------------------------------------------#

export PYTHONPATH=$(pwd)/src:${PYTHONPATH}

DATA_PATH=../../data
PROTO_DUMPS_PATH=$DATA_PATH/proto_dumps
NUM_ITERS=3
LEARNING_RATE=0.1
L2_COEFF=0.1
OUTPUT_PATH=output

rm -r $OUTPUT_PATH 2>/dev/null
mkdir $OUTPUT_PATH

python -u src/learning_baseline/feature_based/train.py \
  --input-train $DATA_PATH/train-v1.0.json \
  --input-train-articles $PROTO_DUMPS_PATH/train-anotatedpartial/train-annotatedpartial.proto \
  --input-train-features $PROTO_DUMPS_PATH/train-feature/train-featuresbucketized.proto \
  --input-dev $DATA_PATH/dev-v1.0.json \
  --input-dev-articles $PROTO_DUMPS_PATH/dev-anotatedpartial/dev-annotatedpartial.proto \
  --input-dev-features $PROTO_DUMPS_PATH/dev-feature/dev-featuresbucketized.proto \
  --model-output $OUTPUT_PATH/model \
  --metrics-output $OUTPUT_PATH/metrics \
  --dev-predictions-output $OUTPUT_PATH/dev-predictions \
  --num-iterations $NUM_ITERS \
  --learning-rate $LEARNING_RATE \
  --l2 $L2_COEFF \
  --input-featuredict $PROTO_DUMPS_PATH/train-feature/featuredictbucketized.proto \
  --use-full-dictionary
