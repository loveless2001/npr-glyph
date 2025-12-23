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
base_model="$HOME/projects/glyph_npr/models/Qwen/Qwen-4B-Instruct-2507"
# Optimized for A100 80GB
# Training config
epochs=1
# Original used 32k context. On single A100 80GB with FlashAttn:
# - BS=1, Seq=32k fits easily.
# - BS=8, Seq=32k might be tight even on 80GB without FSDP sharding across devices.
# We set BS=4 and GradAccum=2 to achieve GlobalBS=8 (matching original 8-GPU setup)
micro_batch_size=4
gradient_accumulation_steps=2
lr=1e-6
weight_decay=0.1

# Run with python directly or torchrun with nproc=1
# using torchrun for consistency with original script but for single GPU
torchrun --nproc-per-node 1 --master_port 23456 train/npr_warmup.py \
    --do_train=True \
    --do_eval=False \
    --block_size=32768 \
    --max_seq_length=32768 \
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
    --attn_implementation="flash_attention_2" \
    --use_liger_kernel=True \
    --save_strategy="no" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --lr_scheduler_type="cosine" \
    --warmup_ratio=0.05 \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --tf32=True
