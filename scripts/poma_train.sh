#!/bin/bash

cd ..

# custom config
DATA=./data
TRAINER=POMA

DATASET=$1
CFG=$2  # config file
CTP=$3  # class token position (end or middle)
NCTX=$4  # number of context tokens
SHOTS=$5  # number of shots (1, 2, 4, 8, 16)
BLOCK=$6  # number of blocks
CSC=$7  # class-specific context (False or True)
CLASS_EQULE=$8  # CLASS_EQULE True of False
TAG=$9 # log tag (multiple_models_random_init or rn50_random_init)
FP=${10} # number of false positive training samples per class

for SEED in {1..3}
do  
    DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLASS_EQULE}_${CONF_THRESHOLD}_${BLOCK}block_${TAG}/nctx${NCTX}_csc${CSC}_ctp${CTP}_fp${FP}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}. Skip this job"
        rm -rf ${DIR}
    else
        echo "Run this job and save the output to ${DIR}"
        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --dataset-config-file configs/datasets/${DATASET}.yaml \
        --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
        --output-dir ${DIR} \
        --num-fp ${FP} \
        TRAINER.POMA.N_CTX ${NCTX} \
        TRAINER.POMA.CSC ${CSC} \
        TRAINER.POMA.CLASS_TOKEN_POSITION ${CTP} \
        TRAINER.POMA.N_BLOCK ${BLOCK} \
        DATASET.NUM_SHOTS ${SHOTS} \
        DATASET.CLASS_EQULE ${CLASS_EQULE}
    fi
done