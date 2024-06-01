#!/bin/bash

DATASET=$1

# Experiments: Abalation Study for DPL
# Configuration
# --- weight parameter beta: 0.6 | 0.7 | 0.8 | 0.9 | 1.0 
# --- dataset: Dtd
# --- noise rate: 0 | 12.5% | 25% | 50%
# --- backbone: Text: ViT-B/32-PT, Visual: RN50-PT
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.6 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.6 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.6 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.6 8

CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.7 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.7 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.7 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.7 8

CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.8 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.8 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.8 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.8 8

CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.9 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.9 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.9 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.9 8

CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 1.0 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 1.0 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 1.0 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 1.0 8

# Experiments: Abalation Study for DPL
# Configuration
# --- prompt blocks m: 1 | 6
# --- dataset: Dtd
# --- noise rate: 0 | 12.5% | 25% | 50%
# --- backbone: Text: ViT-B/32-PT, Visual: RN50-PT
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init 0.5 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init 0.5 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init 0.5 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init 0.5 8

CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init 0.5 0
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init 0.5 2
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init 0.5 4
CUDA_VISIBLE_DEVICES=1 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init 0.5 8

# Experiments: Result Analysis for DPL
# Configuration
# --- Experiments: Abalation Study for DPL -- weight parameter
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.6
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.7
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.8
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.9
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 1.0

# Configuration
# --- Experiments: Abalation Study for DPL -- prompt blocks
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 1 False True rn50_random_init 0.5
CUDA_VISIBLE_DEVICES=1 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 6 False True rn50_random_init 0.5
