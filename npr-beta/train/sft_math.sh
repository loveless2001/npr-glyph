#!/usr/bin/env bash

export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TRITON_CACHE_DIR="/tmp/triton_cache"

export TOKENIZERS_PARALLELISM=false

# if not existing /tmp/triton_cache, create it first
if [ ! -d "$TRITON_CACHE_DIR" ]; then
    mkdir -p "$TRITON_CACHE_DIR"
fi

uid="$(date +%Y%m%d-%H%M%S)"
#base_model="Qwen/Qwen3-4B"
base_model="Qwen/Qwen3-4B-Instruct-2507"

lr=1e-6
min_lr=5e-7
weight_decay=0.1  # -> the same training pipe as slurm_training
gradient_accumulation_steps=1  # requires more GPU memory

epochs=1
micro_batch_size=1  # -> batch_size will be 16 if 16 gpus
gpu_count=$(nvidia-smi -L | wc -l)
world_size=${gpu_count}

torchrun --nproc-per-node ${gpu_count} --master_port 12345 train/npr_warmup.py \
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
    --include_num_input_tokens_seen=True \
    --average_tokens_across_devices=True \
    --dataloader_drop_last=False \
    --seed=521 \
    --data_seed=521 \
    --warmup_ratio=0.05 \
    --bf16=True \
    --tf32=True \
    --bf16_full_eval=True \
    --ddp_backend="nccl" \
    --fsdp="full_shard auto_wrap" \
    --fsdp_config="train/fsdp_config_qwen.json" \
    --metric_for_best_model="eval_pass@1" \
    --greater_is_better=True \
    --eval_on_start=False \
    --eval_strategy="steps" \
    --eval_steps=100000 \
    --logging_steps=1 \
    --save_strategy="steps" \
    --save_steps=1000 \
    --save_total_limit=30 \
    --save_only_model=True \
    --save_safetensors=True \
    --lr_scheduler_type="cosine" \
    --learning_rate=${lr} \
    --weight_decay=${weight_decay} \
    --adam_beta1=0.9 \
    --adam_beta2=0.95 \
    --output_dir="ckpts/NPR-Warmup-4B-Inst-${uid}" \
    --logging_first_step=True \
    --logging_dir="ckpts/NPR-Warmup-4B-Inst-${uid}" \
    --use_liger_kernel=True \
    --disable_tqdm=False \
    --report_to "wandb" \
    --wandb_name="NPR-Warmup-4B-Inst"
