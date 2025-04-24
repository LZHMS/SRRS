#!/bin/bash

DATASET=$1
TYPE=$2

SIR=output/${TYPE}/${DATASET}
DIR=output/${TYPE}/${DATASET}/parsing_results.log

python parse_test_res.py --directory ${SIR} --multi-exp --test-log > ${DIR}