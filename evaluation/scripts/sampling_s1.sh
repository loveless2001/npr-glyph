#!/usr/bin/env bash

# ⚠️ "/share/nlp/share/plm/checkpoint/dapo-RL-zero-4B-v8-70"
# ⚠️ "/share/nlp/share/plm/checkpoint/dapo-RL-zero-4B-v9-90"
# ⚠️ "/share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v11-90"
# ⚠️ "/share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v10-140"
# ✅ "/share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v3-130" (RL-no-think ckpt)
# ✅ "/share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v3-Ins-90" (Instruct ckpt)

set -ex

#export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TP_SIZE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
#MODEL_PATH="/share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v3-130"
#OUTPUT_DIR="output/rejection_sampling_s1_dapo-RL-zero-4B-v3-130_new"
MODEL_PATH="/share/nlp/wutong1/hf_models/dapo-RL-zero-4B-v3-Ins-90"
#OUTPUT_DIR="output/rejection_sampling_s1_dapo-RL-zero-4B-v3-Ins-90_megascience"
OUTPUT_DIR="output/rejection_sampling_s1_dapo-RL-zero-4B-v3-Ins-90_polaris"

PYTHON_BIN=/share/nlp/wutong1/anaconda3/envs/verl/bin/python

${PYTHON_BIN} -u rejection_sampling.py \
    --model_path ${MODEL_PATH} \
    --dataset "Polaris-53K" \
    --instruction "prompts/v2.txt" \
    --output_dir ${OUTPUT_DIR} \
    --tp_size ${TP_SIZE} \
    --max_problems 65536 \
    --batch_size 1024 \
    --min_sample_trial 1 \
    --max_sample_trial 8 \
    --save_interval 1 \
    --max_trajectory_per_problem 1 \
    --max_new_tokens 32768 \
    --temperature 1.0 \
    --top_p 1.0 \
    --top_k -1 \
    --repetition_penalty 1.0 \
    --log_samples 1 \
    --debug \
    --resume_from_checkpoint ${OUTPUT_DIR} \
    --structured_trace_distil
