#!/usr/bin/env bash
net='densenet169'
net_name='baseline'
CUDA_VISIBLE_DEVICES=5 python -u baseline.py \
    --batch_size 8 --epoch 50 --learning_rate 0.0001 --drop_rate 0 \
    --load_model True --load_path ./save/$net/${net_name}.pkl \
    --net_name $net_name \
    --save_path ./save/$net/ \
    > result/$net/${net_name}.out
