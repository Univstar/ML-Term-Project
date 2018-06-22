#!/usr/bin/env bash
net='densenet169'
net_name='densenet169_b8_lr1e-4_d0'
CUDA_VISIBLE_DEVICES=0,2,3,4,5,6 python -u densenet.py \
    --batch_size 8 --epoch 40 --learning_rate 0.0001 --drop_rate 0 \
    --load_model False  \
    --net_name $net_name \
    --save_path save/$net/ \
    > result/$net/$net_name.out