#!/bin/bash

# Read master address and port from the shared file
export MASTER_ADDR=$(awk -F': ' 'NR==1 {print $2}' shared_file.txt)
export MASTER_PORT=$(awk -F': ' 'NR==2 {print $2}' shared_file.txt)
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

# Setup environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NODE_RANK=1
export NUM_NODES=2
export NUM_GPUS_PER_NODE=2
export WORLD_SIZE=4

# Run python script
python single_and_multi_node_multi_gpu.py \
    --node_rank $NODE_RANK \
    --num_nodes $NUM_NODES \
    --num_gpus_per_node $NUM_GPUS_PER_NODE \
    --world_size $WORLD_SIZE \
    --backend "nccl" \
    --init_method "env://" \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    --seed 0 \
    --num_samples 2048 \
    --num_dimensions 20 \
    --target_dimensions 1 \
    --num_workers 0 \
    --pin_memory \
    --sampler_shuffle \
    --model_name toy_model \
    --input_dim 20 \
    --output_dim 1 \
    --criterion_name "mse_loss" \
    --reduction "mean" \
    --optimizer_name "sgd" \
    --lr 1e-3 \
    --max_epochs 50 \
    --save_checkpoint_interval_epoch 10 \
    --batch_size 32 \
    --scheduler_name constant_lr
