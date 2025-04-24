#!/bin/bash

# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh oxford_flowers 12 symflip False 2clip
# CUDA_VISIBLE_DEVICES=0 bash scripts/parse.sh dtd increase

CUDA_VISIBLE_DEVICES=1 bash scripts/common/train.sh caltech101 2 symflip False analysis BLIPScore
CUDA_VISIBLE_DEVICES=2 bash scripts/common/train.sh caltech101 2 pairflip False analysis CoOpLoss
# CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 2 symflip False ablation CoOpMD
# CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 4 symflip False ablation CoOpMD
# CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 6 symflip False ablation CoOpMD
# CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 8 symflip False ablation CoOpMD
# CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 10 symflip False ablation CoOpMD
# CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 12 symflip False ablation CoOpMD
CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 2 pairflip False ablation CoOpMD
CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 4 pairflip False ablation CoOpMD
CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 6 pairflip False ablation CoOpMD
CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 8 pairflip False ablation CoOpMD
CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 10 pairflip False ablation CoOpMD
CUDA_VISIBLE_DEVICES=0 bash scripts/common/train.sh dtd 12 pairflip False ablation CoOpMD
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh dtd 2 symflip False increaseBestEpoch
# CUDA_VISIBLE_DEVICES=0 bash scripts/train2.sh oxford_flowers 12 symflip False no_regular_ep100
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 2 symflip False chosen
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh dtd 12 symflip False chosen
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 8 symflip False chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh dtd 6 symflip False chosen
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 8 symflip False chosen
# CUDA_VISIBLE_DEVICES=0 bash scripts/train2.sh dtd 12 symflip False new_eval
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh eurosat 2 symflip False result 
# CUDA_VISIBLE_DEVICES=0 bash scripts/parse.sh dtd result 

# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh eurosat 2 symflip False ablation
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh eurosat 12 pairflip False ablation
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 12 pairflip False ablation
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh oxford_flowers 4 pairflip False result
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh oxford_flowers 12 symflip False ablation
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh dtd 8 symflip False result

# nohup sh CUDA_VISIBLE_DEVICES=0 bash scripts/train2.sh oxford_flowers 2 symflip False no_regular_ep100 > output0.log 2>&1 &
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh oxford_flowers 2 symflip False no_regular
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh dtd 12 symflip False no_regular

# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 0 pairflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 2 pairflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 4 pairflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 6 pairflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 8 pairflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 10 pairflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh dtd 12 pairflip False

# CUDA_VISIBLE_DEVICES=0 bash scripts/parse.sh dtd dynamic_thres_ep100

# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 0 symflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 2 symflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 4 symflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 6 symflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 8 symflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 10 symflip False
# CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh sun397 12 symflip False
