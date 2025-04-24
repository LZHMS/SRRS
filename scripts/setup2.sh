#!/bin/bash

# nohup sh scripts/setup2.sh > output2.log 2>&1 &

# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 2 symflip False ablation CoOpDN
# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 4 symflip False ablation CoOpDN
# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 6 symflip False ablation CoOpDN
# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 8 symflip False ablation CoOpDN
# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 10 symflip False ablation CoOpDN
# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 12 symflip False ablation CoOpDN
CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 2 pairflip False ablation CoOpDN
CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 4 pairflip False ablation CoOpDN
CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 6 pairflip False ablation CoOpDN
CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 8 pairflip False ablation CoOpDN
CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 10 pairflip False ablation CoOpDN
CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 12 pairflip False ablation CoOpDN

# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 2 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 4 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 6 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 2 symflip False ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 8 pairflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 12 pairflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=2 bash scripts/train2.sh dtd 12 pairflip False ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=2 bash scripts/parse.sh dtd ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 2 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 2 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 4 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 6 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 8 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 12 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 12 symflip False Chosen
# CUDA_VISIBLE_DEVICES=2 bash scripts/parse.sh eurosat Chosen

# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 2 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 2 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 4 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 6 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 8 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 12 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh eurosat 12 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/parse.sh eurosat

# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 2 symflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 2 symflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 4 symflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 6 symflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 8 symflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 12 symflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 12 symflip False

# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 2 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 2 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 4 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 6 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 8 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 12 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/train.sh caltech121 12 pairflip False
# CUDA_VISIBLE_DEVICES=2 bash scripts/parse.sh caltech121