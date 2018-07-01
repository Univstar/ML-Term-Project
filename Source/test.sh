#!/usr/bin/env bash
net='densenet169'
net_name='baseline'
CUDA_VISIBLE_DEVICES=2 python -u densenet.py \
    --batch_size 8 --epoch 0 --learning_rate 0.0001 --drop_rate 0 \
    --load_model False \
    --net_name $net_name \
    --save_path ./save/$net/ \
    > output.out