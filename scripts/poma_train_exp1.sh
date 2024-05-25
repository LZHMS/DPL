#!/bin/bash
: '''
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_Vs 0
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_Vs 2
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_Vs 4
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_Vs 8

CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_Vs 0
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_Vs 2
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_Vs 4
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_Vs 8
'''

CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_VsAd 0
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_VsAd 2
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_VsAd 4
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 1 False True rn50_random_init_VsAd 8

CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_VsAd 0
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_VsAd 2
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_VsAd 4
CUDA_VISIBLE_DEVICES=1 bash poma_train.sh ssdtd rn50_ep50 end 16 16 6 False True rn50_random_init_VsAd 8

