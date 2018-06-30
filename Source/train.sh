#!/usr/bin/env bash
net='densenet169'
net_name='b8_lr1e-4_d0_crop'
CUDA_VISIBLE_DEVICES=5 python -u densenet.py \
    --batch_size 8 --epoch 0 --learning_rate 0.0001 --drop_rate 0 \
    --load_model True --load_path ./save/$net/mura3crop_1_loss.pkl \
    --net_name $net_name \
    --save_path ./save/$net/ \
    > result/$net/${net_name}_+1.out
