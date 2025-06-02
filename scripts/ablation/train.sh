#!/bin/bash

# custom config
DATA=/data1/zhli/SRRS/data
TRAINER=SRRS

DATASET=$1
FP=$2 # number of false positive training samples per class
FPTYPE=$3  #type of noise(symflip, pairflip)
GCE=$4  # GCE loss (False or True)
TYPE=$5
TAG=$6
CFG=rn200  # config file
CTP=end  # class token position (end or middle)
NCTX=16  # number of context tokens
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)

for SEED in 1 2 3
do
    DIR=output_${TYPE}/${TAG}/${DATASET}/${CFG}_${SHOTS}shots_${FP}FP_${FPTYPE}/nctx${NCTX}_csc${CSC}_ctp${CTP}_gce${GCE}/seed${SEED}
    PYDIR=${TYPE}/${TAG}.py
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
        #rm -rf ${DIR}
    else
        echo "Run this job and save the output to ${DIR}"
        python ${PYDIR} \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        TRAINER.SRRS.N_CTX ${NCTX} \
        TRAINER.SRRS.CSC ${CSC} \
        TRAINER.SRRS.GCE ${GCE} \
        TRAINER.SRRS.CLASS_TOKEN_POSITION ${CTP} \
        DATASET.NUM_FP ${FP} \
        DATASET.FP_TYPE ${FPTYPE} \
        DATASET.NUM_SHOTS ${SHOTS}
    fi
done
