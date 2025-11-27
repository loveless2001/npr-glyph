#!/usr/bin/env bash

set -ex

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
TP_SIZE=`nvidia-smi --query-gpu=name --format=csv,noheader | wc -l`
MODEL_PATH="NPR-Warmup-4B-Inst"
OUTPUT_DIR="output/rejection_sampling_NPR-Warmup-4B-Inst"

PYTHON_BIN=python

${PYTHON_BIN} -u sampling/rejection_sampling.py \
    --model_path ${MODEL_PATH} \
    --dataset "ORZ-MATH-57K" \
    --instruction "prompts/npr.txt" \
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
