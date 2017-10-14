#!/bin/bash

#-------------------------------------REQUIRES:--------------------------------#
#                                                                              #
# 1. Training and dev feature files in PROTO_DUMPS_PATH folder                 #
# 2. SQuAD v1.0 train and dev json files                                       #
#                                                                              #
#------------------------------------------------------------------------------#

export PYTHONPATH=$(pwd)/src:${PYTHONPATH}

python -u src/non_learning_baseline/sliding_window.py
