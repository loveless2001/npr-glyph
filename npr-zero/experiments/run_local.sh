#!/bin/bash
set -e

# Environment variables for single-node execution
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=12355
export LOCAL_WORLD_SIZE=1
export LOCAL_RANK=0
export CUDA_VISIBLE_DEVICES=0

# Ensure python path includes current directory
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run the local PPO training script
# Using a dummy config or the user's config
# Note: You might need to adjust config path and name
# Run the local PPO training script
# Using the user's Qwen-4B configuration
# Note: Ensure you have downloaded the model to the specified path or update it.
# Run the local PPO training script
# Using the user's Qwen-4B configuration
# Optimized for A100 80GB: Full Fine-Tuning (No LoRA), Larger Batches
python verl/trainer/main_ppo_local.py \
    data.train_files=$HOME/projects/glyph_npr/npr-zero/data/train.parquet \
    data.val_files=$HOME/projects/glyph_npr/npr-zero/data/test.parquet \
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
    +trainer.experiment_name=npr_zero_local_zero \
    +prioritize_flash_attention=True
    # LoRA disabled for Full Fine-Tuning on 80GB card
    # +actor_rollout_ref.model.lora_rank=64 \
    # +actor_rollout_ref.model.lora_alpha=32 \
    # +actor_rollout_ref.model.target_modules=[q_proj,v_proj] \
