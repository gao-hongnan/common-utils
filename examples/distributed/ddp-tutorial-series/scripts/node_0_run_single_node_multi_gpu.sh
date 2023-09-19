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
export NUM_NODES=1
export NUM_GPUS_PER_NODE=4
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

# Compare the last two lines of log files
# Define a function to compare the last two lines of log files
# Modify compare_logs to handle stripping and aggregation logic
compare_logs() {
    local current_log=$1
    local ground_truth_log=$2
    local node_count=$3
    local gpu_count=$4

    if [ "$node_count" -eq 1 ] && [ "$gpu_count" -gt 1 ]; then
        # Strip off "Node X GPU X" from the log
        local current_last_lines=$(tail -n 2 $current_log | sed 's/^[^[]*//' | sed 's/\[\(INFO\|ERROR\|DEBUG\|WARN\|WARNING\|CRITICAL\)\]: \[\(TRAIN\|VALID\): NODE[0-9]\+ GPU[0-9]\+\] //')
        local ground_truth_last_lines=$(tail -n 2 $ground_truth_log | sed 's/^[^[]*//' | sed 's/\[\(INFO\|ERROR\|DEBUG\|WARN\|WARNING\|CRITICAL\)\]: \[\(TRAIN\|VALID\): NODE[0-9]\+ GPU[0-9]\+\] //')
    else
        local current_last_lines=$(tail -n 2 $current_log | sed 's/^[^[]*//')
        local ground_truth_last_lines=$(tail -n 2 $ground_truth_log | sed 's/^[^[]*//')
    fi

    # Compare
    if [ "$current_last_lines" != "$ground_truth_last_lines" ]; then
        echo "Difference detected in the last two lines of $current_log compared to the ground truth!"
        echo "Since we are training on single node, the difference could be just the node rank."
        echo "It could also be due to global rank 0 having more logs than other global ranks."
        echo "Current last two lines:"
        echo "$current_last_lines"
        echo "Ground truth last two lines:"
        echo "$ground_truth_last_lines"
    else
        echo "Last two lines of $current_log match with the ground truth."
    fi
}

# If only one node and one GPU, aggregate and divide by 4
if [ "$NUM_NODES" -eq 1 ] && [ "$NUM_GPUS_PER_NODE" -eq 1 ]; then
    compare_logs process_all_reduce.log ./tests/ground_truths/single_and_multi_node_multi_gpu/process_all_reduce.txt $NUM_NODES $NUM_GPUS_PER_NODE
else
    # Iterate over logs and compare
    for i in 0 1 2 3; do
        CURRENT_LOG="process_${i}.log"
        GROUND_TRUTH_LOG="./tests/ground_truths/single_and_multi_node_multi_gpu/process_${i}.txt"

        compare_logs $CURRENT_LOG $GROUND_TRUTH_LOG $NUM_NODES $NUM_GPUS_PER_NODE
    done
fi
