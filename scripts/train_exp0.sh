#!/bin/bash

DATASET=$1

# Experiments: Training for DPL
# Configuration
# --- dataset: Dtd
# --- noise rate: 0 | 12.5% | 25% | 50%
# --- backbone: Text: ViT-B/32-PT, Visual: RN50-PT 
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 8


# Experiments: Abalation Study for DPL
# Configuration
# --- weight parameter beta: 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5
# --- dataset: Dtd
# --- noise rate: 0 | 12.5% | 25% | 50%
# --- backbone: Text: ViT-B/32-PT, Visual: RN50-PT
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.0 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.0 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.0 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.0 8

CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.1 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.1 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.1 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.1 8

CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.2 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.2 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.2 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.2 8

CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.3 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.3 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.3 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.3 8

CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.4 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.4 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.4 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.4 8

CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 8

# Experiments: Abalation Study for DPL
# Configuration
# --- prompt blocks m: 2 | 4
# --- dataset: Dtd
# --- noise rate: 0 | 12.5% | 25% | 50%
# --- backbone: Text: ViT-B/32-PT, Visual: RN50-PT
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init 0.5 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init 0.5 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init 0.5 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init 0.5 8

CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 0
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 2
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 4
CUDA_VISIBLE_DEVICES=0 bash dpl_train.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5 8

# Experiments: Result Analysis for DPL
# Configuration
# --- Experiments: Training for DPL
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5

# Configuration
# --- Experiments: Abalation Study for DPL -- weight parameter
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.0
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.1
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.2
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.3
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.4
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 4 False True rn50_random_init 0.5

# Configuration
# --- Experiments: Abalation Study for DPL -- prompt blocks
CUDA_VISIBLE_DEVICES=0 bash parse_test.sh ${DATASET} rn50_ep50 end 16 16 2 False True rn50_random_init 0.5