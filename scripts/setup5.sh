#!/bin/bash

CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 0 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 2 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 4 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 6 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 8 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 10 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/train3.sh dtd 12 symflip False ablation MixDivideWarmupAugmentationBLIP
CUDA_VISIBLE_DEVICES=3 bash scripts/parse.sh dtd ablation MixDivideWarmupAugmentationBLIP