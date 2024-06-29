#!/bin/bash

export ROOT_DIR1=/data/dataset/nerf_wild/brandenburg_gate/dense
export ROOT_DIR2=/data/dataset/nerf_wild/sacre_coeur/dense
export ROOT_DIR3=/data/dataset/nerf_wild/trevi_fountain/dense

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --source_path $ROOT_DIR3 \
#     --exp_name trevi_preenc_mask_mlp4_64_ssim_half --iterations 30_000 --convert_SHs_python \
#     --resolution 2 --min_opacity 0.005 \
#     --images images --mlp_width 64 \
#     --densify_until_iter 15_000 \
#     --densify_grad_threshold 0.0002 --a_enc --mask_unet

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --source_path $ROOT_DIR1 \
#     --exp_name gate_baseline_enca_nopre --iterations 30_000 --convert_SHs_python \
#     --resolution 2 --min_opacity 0.005 \
#     --images images \
#     --densify_until_iter 15_000 \
#     --densify_grad_threshold 0.0002 --a_enc

# CUDA_VISIBLE_DEVICES=0 \
# python train.py \
#     --source_path $ROOT_DIR2 \
#     --exp_name sacre_baseline_enca_nopre --iterations 30_000 --convert_SHs_python \
#     --resolution 2 --min_opacity 0.005 \
#     --images images \
#     --densify_until_iter 15_000 \
#     --densify_grad_threshold 0.0002 --a_enc


CUDA_VISIBLE_DEVICES=0 \
python train.py \
    --source_path $ROOT_DIR3 \
    --exp_name trevi_preenc_mask_di_2 --iterations 30_000 --convert_SHs_python \
    --resolution 2 --min_opacity 0.005 \
    --images images \
    --densify_until_iter 15_000 \
    --densify_grad_threshold 0.0002 --a_enc --mask_unet --maskrs_k 0.00012