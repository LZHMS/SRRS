#!/bin/bash


# CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 2 symflip False ablation CoOpDC
# CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 4 symflip False ablation CoOpDC
# CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 6 symflip False ablation CoOpDC
# CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 8 symflip False ablation CoOpDC
# CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 10 symflip False ablation CoOpDC
# CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 12 symflip False ablation CoOpDC
CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 2 pairflip False ablation CoOpDC
CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 4 pairflip False ablation CoOpDC
CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 6 pairflip False ablation CoOpDC
CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 8 pairflip False ablation CoOpDC
CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 10 pairflip False ablation CoOpDC
CUDA_VISIBLE_DEVICES=1 bash scripts/ablation/train.sh dtd 12 pairflip False ablation CoOpDC

# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 2 symflip False ablation CoOpDC
# CUDA_VISIBLE_DEVICES=2 bash scripts/ablation/train.sh dtd 2 symflip False ablation CoOpDN

# CUDA_VISIBLE_DEVICES=1 bash scripts/train3.sh oxford_flowers 2 symflip False no_regular_ep211
# CUDA_VISIBLE_DEVICES=1 bash scripts/train_base.sh eurosat 11 pairflip False test
#CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 4 symflip False result
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 6 symflip False result
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 8 symflip False result

# CUDA_VISIBLE_DEVICES=1 bash scripts/train2.sh oxford_flowers 1 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=1 bash scripts/train2.sh oxford_flowers 2 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=1 bash scripts/train2.sh oxford_flowers 4 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=1 bash scripts/train2.sh oxford_flowers 6 symflip False ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=1 bash scripts/parse.sh oxford_flowers ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 1 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 2 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 4 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 6 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 8 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 11 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh dtd 12 pairflip False ablation MixDivideWarmupAugmentationRegularFPL
# CUDA_VISIBLE_DEVICES=1 bash scripts/parse.sh dtd ablation MixDivideWarmupAugmentationRegularFPL

# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 1 symflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 2 symflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 4 symflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 6 symflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 8 symflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 11 symflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 12 symflip False

# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 1 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 2 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 4 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 6 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 8 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 11 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh ucf111 12 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/parse.sh ucf111

# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 1 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 2 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 4 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 6 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 8 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 11 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/train.sh sun397 12 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/parse.sh sun397

# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 1 symflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 2 symflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 4 symflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 6 symflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 8 symflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 11 symflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 12 symflip False

# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 1 pairflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 2 pairflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 4 pairflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 6 pairflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 8 pairflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 11 pairflip False
# CUDA_VISIBLE_DEVICES=1,1 bash scripts/train.sh stanford_cars 12 pairflip False
# CUDA_VISIBLE_DEVICES=1 bash scripts/parse.sh stanford_cars
