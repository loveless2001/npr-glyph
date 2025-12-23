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
python recipe/dapo/main_dapo_local.py \
    data.train_files=$HOME/projects/glyph_npr/npr-rl/examples/data/math_total/train.parquet \
    data.val_files=$HOME/projects/glyph_npr/npr-rl/examples/data/math_total/test.parquet \
    actor_rollout_ref.model.path=$HOME/projects/glyph_npr/models/Qwen/Qwen1.5-0.5B-Chat \
    actor_rollout_ref.actor.strategy=fsdp \
    critic.strategy=fsdp \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    +trainer.experiment_name=npr_rl_local_test
