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
 
DIR=./output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots_EQULE_${CLASS_EQULE}_${CONF_THRESHOLD}_${BLOCK}block_${TAG}/nctx${NCTX}_csc${CSC}_ctp${CTP}_fp${FP}/
python parse_test_res.py --directory ${DIR} --test-log --ci95