#!/usr/bin/env bash

set -xeuo pipefail

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python filtering.py \
    --model_path /share/nlp/share/plm/Qwen3-4B \
    --data_path /share/nlp/liuyang/workspace/parallel_reasoner/sl/dataset/math/rejection_sampling_rl_ORZ-MATH-57K_20250921T094655.json \
    --protocol prompts/parallel_logic_scoring_en_v0.md \
    --output_dir output/refined \
    --repeat_times 1 \
    --max_new_tokens 8192 \
    --temperature 0.6
