#!/usr/bin/env bash`
set -xeuo pipefail
unset ROCR_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

export NCCL_IBEXT_DISABLE=1
export NCCL_NVLS_ENABLE=1
export NCCL_IB_HCA=mlx5
export UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1,mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1

project_name='math-reason'
exp_name='RL-PA-4B-fromIns'

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.2

max_prompt_length=$((1024 * 2))
max_response_length=30000
enable_overlong_buffer=False
overlong_buffer_len=10
overlong_penalty_factor=1.0
overlong_buffer_log=True

loss_agg_mode="token-mean"

enable_filter_groups=False
filter_groups_metric=score
max_num_gen_batches=0
train_prompt_bsz=16
gen_prompt_bsz=16
n_resp_per_prompt=16
train_prompt_mini_bsz=1

# Ray
# RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:2333"}
# WORKING_DIR=${WORKING_DIR:-"${PWD}"}
# RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=1
# Paths
RAY_DATA_HOME=${RAY_DATA_HOME:-"<Your-Absolute-Path>"}
MODEL_PATH=${MODEL_PATH:-"<Your-Model-Path>"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/experiments/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/experiments/data/math_total/train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/experiments/data/math_total/test.parquet"}
VALID_DATA_DIR=${VALID_DATA_DIR:-"${RAY_DATA_HOME}/experiments/${project_name}_logs/${exp_name}"}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
val_temperature=1.0

GPU_NUM=8

# Performance Related Parameter
sp_size=1
# use_dynamic_bsz=True
# actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
# infer_ppo_max_token_len=$((2 * (max_prompt_length + max_response_length)))
ppo_micro_batch_size_per_gpu=1
log_prob_micro_batch_size_per_gpu=1
offload=True
gen_tp=4

ray_tmp_dir=ray_tmp

ray stop --force 2>/dev/null || true
unset RAY_ADDRESS RAY_REDIS_ADDRESS RAY_JOB_ID
ray start --head --port=8964 --num-cpus=16 --temp-dir=${ray_tmp_dir} --disable-usage-stats

python3 -m recipe.dapo.main_dapo \
    ray_tmp_dir=${ray_tmp_dir} \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.shuffle=False \
    data.dataloader_num_workers=32 \
    data.filter_overlong_prompts=true \
    data.filter_overlong_prompts_workers=32 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.gen_batch_size=${gen_prompt_bsz} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${ppo_micro_batch_size_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${log_prob_micro_batch_size_per_gpu} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.offload_policy=False \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.optim.min_lr_ratio=1e-8 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    reward_model.overlong_buffer.log=${overlong_buffer_log} \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${GPU_NUM}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.balance_batch=False \
    trainer.test_freq=5 \
    trainer.save_freq=5 \
    trainer.total_epochs=1 \
    trainer.validation_data_dir="${VALID_DATA_DIR}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    # trainer.run_id=
