#!/usr/bin/env bash
net='densenet201'
net_name='b8_lr1e-4_d0_logloss'
CUDA_VISIBLE_DEVICES=3 python -u densenet201.py \
    --batch_size 8 --epoch 50 --learning_rate 0.0001 --drop_rate 0 \
    --loss_type logloss \
    --load_model False \
    --net_name $net_name --save_path ./save/$net/ \
    > result/$net/${net_name}.out &

net='densenet201'
net_name='b8_lr1e-4_d0_focalloss'
CUDA_VISIBLE_DEVICES=4 python -u densenet201.py \
    --batch_size 8 --epoch 50 --learning_rate 0.0001 --drop_rate 0 \
    --loss_type focalloss \
    --load_model False \
    --net_name $net_name --save_path ./save/$net/ \
    > result/$net/${net_name}.out