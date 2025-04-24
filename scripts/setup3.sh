#!/bin/bash

# nohup sh scripts/setup3.sh > output3.log 3>&1 &

# CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 2 symflip False ablation CoRegMD
# CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 4 symflip False ablation CoRegMD
# CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 6 symflip False ablation CoRegMD
# CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 8 symflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 10 symflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 12 symflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 2 pairflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 4 pairflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 6 pairflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 8 pairflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 10 pairflip False ablation CoRegMD
CUDA_VISIBLE_DEVICES=3 bash scripts/ablation/train.sh dtd 12 pairflip False ablation CoRegMD
# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 8 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 10 symflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 13 symflip False ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 0 pairflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 3 pairflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 4 pairflip False ablation MixDivideWarmupAugmentationBLIP
# CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 6 pairflip False ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=3 bash scripts/parse.sh dtd ablation MixDivideWarmupAugmentationBLIP

# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 0 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 3 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 4 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 6 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 8 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 10 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 13 symflip False

# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 0 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 3 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 4 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 6 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 8 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 10 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_pets 13 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/parse.sh oxford_pets

# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 0 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 3 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 4 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 6 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 8 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 10 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 13 symflip False

# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 0 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 3 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 4 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 6 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 8 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 10 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh oxford_flowers 13 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/parse.sh oxford_flowers

# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 0 symflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh sun397 3 symflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 4 symflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 6 symflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 8 symflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 10 symflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 13 symflip False

# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 0 pairflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 3 pairflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 4 pairflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 6 pairflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 8 pairflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 10 pairflip False
# CUDA_VISIBLE_DEVICES=3,3 bash scripts/train.sh sun397 13 pairflip False
# CUDA_VISIBLE_DEVICES=3 bash scripts/parse.sh sun397

CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 0 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 3 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 4 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 6 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 8 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 10 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/train.sh eurosat 13 pairflip False Chosen
CUDA_VISIBLE_DEVICES=3 bash scripts/parse.sh eurosat Chosen