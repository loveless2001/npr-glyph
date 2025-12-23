#!/bin/bash
set -e

# Local single-node execution script for NPR-RL (DAPO)
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12366
export LOCAL_WORLD_SIZE=1
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Paths
export PYTHON_PATH=$PYTHONPATH:$(pwd)

# Run
# Optimized for A100 80GB: Full Fine-Tuning (No LoRA), Larger Batches
python recipe/dapo/main_dapo_local.py \
    data.train_files=$HOME/projects/glyph_npr/npr-rl/examples/data/math_total/train.parquet \
    data.val_files=$HOME/projects/glyph_npr/npr-rl/examples/data/math_total/test.parquet \
    actor_rollout_ref.model.path=$HOME/projects/glyph_npr/models/Qwen/Qwen-4B-Instruct-2507 \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    critic.strategy=fsdp \
    critic.model.enable_gradient_checkpointing=False \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    data.train_batch_size=256 \
    data.val_batch_size=128 \
    +trainer.experiment_name=npr_rl_local_qwen4b_80gb \
    +prioritize_flash_attention=True
    # LoRA disabled for Full Fine-Tuning
    # +actor_rollout_ref.model.lora_rank=64 \
    # +actor_rollout_ref.model.lora_alpha=32 \
    # +actor_rollout_ref.model.target_modules=[q_proj,v_proj] \
