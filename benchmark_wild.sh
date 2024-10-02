#!/bin/bash

# please change dir to your dataset path
export ROOT_DIR1=/aiarena/gpfs/data/dataset/nerf_wild/brandenburg_gate/dense
export ROOT_DIR2=/aiarena/gpfs/data/dataset/nerf_wild/sacre_coeur/dense
export ROOT_DIR3=/aiarena/gpfs/data/dataset/nerf_wild/trevi_fountain/dense

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --source_path $ROOT_DIR1 \
#     --exp_name brandenburg_gate --iterations 30_000 --convert_SHs_python \
#     --resolution 2 --min_opacity 0.005 \
#     --images images --mlp_width 128 \
#     --densify_until_iter 15_000 \
#     --densify_grad_threshold 0.0002 --a_enc --mask_unet --maskrs_max 0.5 --maskrs_min 0.18


# CUDA_VISIBLE_DEVICES=1 \
# python train.py \
#     --source_path $ROOT_DIR2 \
#     --exp_name sacre_coeur --iterations 30_000 --convert_SHs_python \
#     --resolution 2 --min_opacity 0.005 \
#     --images images --mlp_width 128 \
#     --densify_until_iter 15_000 \
#     --densify_grad_threshold 0.0002 --a_enc --mask_unet --maskrs_max 0.5 --maskrs_min 0.18

CUDA_VISIBLE_DEVICES=2 \
python train.py \
    --source_path $ROOT_DIR3 \
    --exp_name trevi_fountain --iterations 30_000 --convert_SHs_python \
    --resolution 2 --min_opacity 0.005 \
    --images images --mlp_width 128 \
    --densify_until_iter 15_000 \
    --densify_grad_threshold 0.0002 --a_enc --mask_unet --maskrs_max 0.5 --maskrs_min 0.13