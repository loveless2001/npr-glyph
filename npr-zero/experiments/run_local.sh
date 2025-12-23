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
python verl/trainer/main_ppo_local.py \
    data.train_files=$HOME/projects/glyph_npr/npr-zero/data/train.parquet \
    data.val_files=$HOME/projects/glyph_npr/npr-zero/data/test.parquet \
    actor_rollout_ref.model.path=$HOME/projects/glyph_npr/models/Qwen/Qwen1.5-0.5B-Chat \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    +trainer.experiment_name=npr_zero_local_test
