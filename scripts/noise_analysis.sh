#!/bin/bash
cd ..

# custom config
TRAINER=$1
DATASET=$2
CFG=$3  # config file
CTP=$4  # class token position (end or middle)
NCTX=$5  # number of context tokens
SHOTS=$6  # number of shots (1, 2, 4, 8, 16)
CSC=$7  # class-specific context (False or True)
CLSQ=$8
TAG=$9

DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLSQ}__${TAG}/noise_analysis.txt
python noise_analysis.py \
        --trainer ${TRAINER} \
        --dataset ${DATASET} \
        --config ${CFG} \
        --nctx ${NCTX} \
        --ctp ${CTP} \
        --shots ${SHOTS} \
        --clsequle \
        --tag ${TAG} \
        > ${DIR}