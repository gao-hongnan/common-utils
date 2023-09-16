#!/bin/bash

# Get master address and port
export MASTER_ADDR=$(hostname -i)
export MASTER_PORT=$(comm -23 <(seq 1 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"

echo "Master Address: $MASTER_ADDR" > shared_file.txt
echo "Master Port: $MASTER_PORT" >> shared_file.txt

# Setup other environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export NODE_RANK=0
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
    --input_dim 20 \
    --output_dim 1 \
    --criterion_name "mse_loss" \
    --reduction "mean" \
    --optimizer_name "sgd" \
    --lr 1e-3 \
    --max_epochs 50 \
    --save_checkpoint_interval 10 \
    --batch_size 32 \
    --scheduler_name constant_lr

# Compare the last two lines of log files
for i in 0 1 2 3; do
    CURRENT_LOG="process_${i}.log"
    GROUND_TRUTH_LOG="./tests/ground_truths/single_and_multi_node_multi_gpu/process_${i}.txt"

    # Extract the last two lines
    CURRENT_LAST_LINES=$(tail -n 2 $CURRENT_LOG)
    GROUND_TRUTH_LAST_LINES=$(tail -n 2 $GROUND_TRUTH_LOG)

    # Compare
    if [ "$CURRENT_LAST_LINES" != "$GROUND_TRUTH_LAST_LINES" ]; then
        echo "Difference detected in the last two lines of process_${i}.log compared to the ground truth!"
        echo "Current last two lines:"
        echo "$CURRENT_LAST_LINES"
        echo "Ground truth last two lines:"
        echo "$GROUND_TRUTH_LAST_LINES"
    else
        echo "Last two lines of process_${i}.log match with the ground truth."
    fi
done