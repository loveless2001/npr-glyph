#!/usr/bin/env bash

# Local single-node execution script for NPR-Beta (SFT Warmup)
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR="/tmp/triton_cache"
export TOKENIZERS_PARALLELISM=false

# if not existing /tmp/triton_cache, create it first
if [ ! -d "$TRITON_CACHE_DIR" ]; then
    mkdir -p "$TRITON_CACHE_DIR"
fi

uid="$(date +%Y%m%d-%H%M%S)"
base_model="Qwen/Qwen1.5-0.5B-Chat" # Default to smaller model for local testing

# Training config
epochs=1
micro_batch_size=1
gradient_accumulation_steps=4

# Run with python directly or torchrun with nproc=1
# using torchrun for consistency with original script but for single GPU
torchrun --nproc-per-node 1 --master_port 23456 train/npr_warmup.py \
    --do_train=True \
    --do_eval=False \
    --block_size=2048 \
    --max_seq_length=2048 \
    --per_device_train_batch_size=${micro_batch_size} \
    --per_device_eval_batch_size=${micro_batch_size} \
    --gradient_accumulation_steps=${gradient_accumulation_steps} \
    --num_train_epochs=${epochs} \
    --train_file_path="dataset/math/rejection_sampling/train" \
    --valid_file_path="dataset/benchmark/aime25" \
    --test_file_path="dataset/benchmark/aime25" \
    --model_name=${base_model} \
    --include_tokens_per_second=True \
    --dataloader_drop_last=False \
    --output_dir="ckpts/NPR-Warmup-Local-${uid}" \
    --report_to="none" \
    --logging_first_step=True \
    --logging_steps=1 \
    --bf16=True \
    --use_liger_kernel=False \
    --save_strategy="no"
