#!/bin/bash
: '''
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssucf101 rn50_ep50 end 16 16 4 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssucf101 rn50_ep50 end 16 16 4 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssucf101 rn50_ep50 end 16 16 4 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssucf101 rn50_ep50 end 16 16 4 False True rn50_random_init 8
'''
: '''
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_Vs 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_Vs 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_Vs 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_Vs 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_Vs 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_Vs 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_Vs 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_Vs 8
'''
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_VsAd 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_VsAd 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_VsAd 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 2 False True rn50_random_init_VsAd 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_VsAd 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_VsAd 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_VsAd 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssdtd rn50_ep50 end 16 16 4 False True rn50_random_init_VsAd 8

: '''
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh sscaltech101 rn50_ep50 end 16 16 4 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh sscaltech101 rn50_ep50 end 16 16 4 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh sscaltech101 rn50_ep50 end 16 16 4 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh sscaltech101 rn50_ep50 end 16 16 4 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_pets rn50_ep50 end 16 16 4 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_pets rn50_ep50 end 16 16 4 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_pets rn50_ep50 end 16 16 4 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_pets rn50_ep50 end 16 16 4 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_flowers rn50_ep50 end 16 16 4 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_flowers rn50_ep50 end 16 16 4 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_flowers rn50_ep50 end 16 16 4 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssoxford_flowers rn50_ep50 end 16 16 4 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssfgvc_aircraft rn50_ep50 end 16 16 4 False True rn50_random_init 0
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssfgvc_aircraft rn50_ep50 end 16 16 4 False True rn50_random_init 2
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssfgvc_aircraft rn50_ep50 end 16 16 4 False True rn50_random_init 4
CUDA_VISIBLE_DEVICES=0 bash poma_train.sh ssfgvc_aircraft rn50_ep50 end 16 16 4 False True rn50_random_init 8

CUDA_VISIBLE_DEVICES=0 bash noise_analysis.sh POMA ssdtd rn50_ep50 end 16 16 False True 4block_rn50_random_init
CUDA_VISIBLE_DEVICES=0 bash noise_analysis.sh POMA ssucf101 rn50_ep50 end 16 16 False True 4block_rn50_random_init
CUDA_VISIBLE_DEVICES=0 bash noise_analysis.sh POMA sscaltech101 rn50_ep50 end 16 16 False True 4block_rn50_random_init
CUDA_VISIBLE_DEVICES=0 bash noise_analysis.sh POMA ssoxford_flowers rn50_ep50 end 16 16 False True 4block_rn50_random_init
CUDA_VISIBLE_DEVICES=0 bash noise_analysis.sh POMA ssfgvc_aircraft rn50_ep50 end 16 16 False True 4block_rn50_random_init
CUDA_VISIBLE_DEVICES=0 bash noise_analysis.sh POMA ssoxford_pets rn50_ep50 end 16 16 False True 4block_rn50_random_init

'''

