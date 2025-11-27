#!/usr/bin/env bash

set -xeuo pipefail

export HYDRA_FULL_ERROR=1
export RAY_RUNTIME_ENV_TEMPORARY_REFERENCE_EXPIRATION_S=600

# ============================================================================
# Help Function
# ============================================================================

show_help() {
    cat << EOF
Usage: $(basename "$0") [OPTIONS]

Evaluation script for math reasoning models with configurable parameters.

OPTIONS:
  -h, --help                            Show this help message and exit

  Hardware & Parallelization:
  --cuda <devices>              GPU devices to use (default: 0,1,2,3,4,5,6,7)
  --tp_size <n>                         Tensor parallel size (default: 1)
  --dp_size <n>                         Data parallel size (default: auto-detect from nvidia-smi)
  --mem_fraction <f>                    GPU memory fraction to use (default: 0.9)

  Tasks & Dataset:
  --task, --tasks <tasks>               Tasks to evaluate (space-separated): AIME25, AIME24, GPQA (default: AIME25)
  --max_eval_samples, --max_problems <n> Maximum number of problems to evaluate (default: 30000)
  --eval_batch_size, --batch_size <n>   Evaluation batch size (default: 4)

  Model Configuration:
  --model_path <path>                   Path to model checkpoint (default: Qwen/Qwen3-4B)
  --prompt_path, --instruction <path>   Path to prompt/instruction file (default: prompts/npr.txt)
  --engine <engine>                     Inference engine: parallel or sequential (default: parallel)

  Sampling Parameters:
  --num_samples <n>                     Number of samples per problem (default: 8)
  --k, --passk <k>                      Pass@k metric to compute (default: 1)
  --max_new_tokens <n>                  Maximum tokens to generate (default: 30000)
  --temperature <t>                     Sampling temperature (default: 1.0)
  --top_p <p>                           Top-p (nucleus) sampling threshold (default: 0.7)
  --top_k <k>                           Top-k sampling (-1 to disable) (default: -1)

  Output & Logging:
  --log_samples <n>                     Number of samples to log (default: 3)
  --save_total_limit <n>                Maximum number of checkpoints to save (default: 3)

  Flags (presence enables the flag):
  --apply_chat                          Apply chat template to prompts
  --overwrite                           Overwrite existing results
  --debug                               Enable debug mode
  --enable_thinking                     Enable thinking mode for Qwen3 models

EXAMPLES:
  # Basic evaluation on AIME25
  $(basename "$0") --task AIME25 --model_path /path/to/model

  # Multiple tasks with custom sampling
  $(basename "$0") --tasks "AIME25 GPQA" --num_samples 16 --temperature 0.8

  # Specific GPU configuration
  $(basename "$0") --cuda 0,1 --dp_size 2 --batch_size 8

EOF
    exit 0
}

# ============================================================================
# Parse Command Line Arguments
# ============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help ;;
        --cuda) CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --task|--tasks) TASK="$2"; shift 2 ;;
        --eval_batch_size|--batch_size) EVAL_BATCH_SIZE="$2"; shift 2 ;;
        --max_eval_samples|--max_problems) MAX_EVAL_SAMPLES="$2"; shift 2 ;;
        --num_samples) NUM_SAMPLES="$2"; shift 2 ;;
        --k|--passk) K="$2"; shift 2 ;;
        --model_path) MODEL_PATH="$2"; shift 2 ;;
        --prompt_path|--instruction) PROMPT_PATH="$2"; shift 2 ;;
        --tp_size) TP_SIZE="$2"; shift 2 ;;
        --dp_size) DP_SIZE="$2"; shift 2 ;;
        --engine)
            case "$2" in
                parallel) ENGINE="parallel" PARALLEL_REASONING=1 ;;
                sequential) ENGINE="sequential" PARALLEL_REASONING=0 ;;
                multiverse) ENGINE="multiverse" PARALLEL_REASONING=1 ;;
                *) echo "Invalid engine: $2. Must be 'parallel', 'sequential', or 'multiverse'"; exit 1 ;;
            esac
            shift 2 ;;

        --max_new_tokens) MAX_NEW_TOKENS="$2"; shift 2 ;;
        --mem_fraction) MEM_FRACTION="$2"; shift 2 ;;
        --temperature) TEMPERATURE="$2"; shift 2 ;;
        --top_p) TOP_P="$2"; shift 2 ;;
        --top_k) TOP_K="$2"; shift 2 ;;
        --log_samples) LOG_SAMPLES="$2"; shift 2 ;;
        --save_total_limit) SAVE_TOTAL_LIMIT="$2"; shift 2 ;;
        --apply_chat) APPLY_CHAT=1; shift ;;
        --overwrite) OVERWRITE=1; shift ;;
        --debug) DEBUG=1; shift ;;
        --enable_thinking) ENABLE_THINKING=1; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# ============================================================================
# Configuration Options (Defaults)
# ============================================================================

# CUDA devices (uncomment one or set via environment)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# Tasks: AIME25, AIME24, GPQA, or space-separated combination
TASK=${TASK:-"AIME25"}

# Evaluation parameters
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
MAX_EVAL_SAMPLES=${MAX_EVAL_SAMPLES:-30000}
NUM_SAMPLES=${NUM_SAMPLES:-8}
K=${K:-1}

# ============================================================================
# Model Selection (set MODEL_PATH or uncomment one)
# ============================================================================

BASE_QWEN3_4B=Qwen/Qwen3-4B
BASE_QWEN3_4B_BASE=Qwen/Qwen3-4B-Base
BASE_QWEN3_4B_INSTRUCT=Qwen/Qwen3-4B-Instruct-2507

# Active model (set via env or uncomment)
MODEL_PATH=${MODEL_PATH:-$BASE_QWEN3_4B}

# ============================================================================
# Inference Configuration
# ============================================================================

# Prompt: "", prompts/cot.txt, prompts/npr.txt
PROMPT_PATH=${PROMPT_PATH:-prompts/npr.txt}

# Parallelization: tp (tensor parallel), dp (data parallel)
MODE=${MODE:-dp}
TP_SIZE=${TP_SIZE:-1}
DP_SIZE=${DP_SIZE:-$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)}

# Engine: parallel (sglang), sequential (verl)
ENGINE=${ENGINE:-parallel}

# Generation parameters
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-30000}
MEM_FRACTION=${MEM_FRACTION:-0.9}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.7}
TOP_K=${TOP_K:--1}
LOG_SAMPLES=${LOG_SAMPLES:-3}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-3}

# Flags (disabled by default, enabled when specified)
APPLY_CHAT=${APPLY_CHAT:-0}
OVERWRITE=${OVERWRITE:-0}
DEBUG=${DEBUG:-0}
PARALLEL_REASONING=${PARALLEL_REASONING:-0}
ENABLE_THINKING=${ENABLE_THINKING:-0}

# ============================================================================
# Ray Configuration (uncomment to enable)
# ============================================================================
ray stop --force 2>/dev/null || true
#unset RAY_ADDRESS RAY_REDIS_ADDRESS RAY_JOB_ID
ray start --head --port=1234 --temp-dir=/tmp --disable-usage-stats

# ============================================================================
# Execute Evaluation
# ============================================================================

# Map engine names: parallel -> sglang, sequential -> verl
case "$ENGINE" in
    parallel) PYTHON_BIN="python" ;;
    sequential) PYTHON_BIN="python" ;;
    multiverse) PYTHON_BIN="python" ;;
    *) ENGINE_IMPL="$ENGINE" ;;
esac

${PYTHON_BIN} -u evaluate.py \
    --tasks ${TASK} \
    --model_path ${MODEL_PATH} \
    --instruction ${PROMPT_PATH} \
    --tp_size ${TP_SIZE} \
    --dp_size ${DP_SIZE} \
    --max_problems ${MAX_EVAL_SAMPLES} \
    --batch_size ${EVAL_BATCH_SIZE} \
    --num_samples ${NUM_SAMPLES} \
    --passk ${K} \
    --log_samples ${LOG_SAMPLES} \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --mem_fraction_static ${MEM_FRACTION} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --top_k ${TOP_K} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    $([ "$APPLY_CHAT" = "1" ] && echo "--apply_chat") \
    $([ "$OVERWRITE" = "1" ] && echo "--overwrite") \
    $([ "$DEBUG" = "1" ] && echo "--debug") \
    $([ "$PARALLEL_REASONING" = "1" ] && echo "--parallel_reasoning") \
    $([ "$ENABLE_THINKING" = "1" ] && echo "--enable_thinking")
