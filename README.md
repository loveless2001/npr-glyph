# Title


## Introduction
xxxx


## Getting Started
### Stage 1: NPR-Zero

#### How to install
```
cd NPR_Zero
conda create -n zero python=3.11
conda activate zero
conda install nvidia::cuda-nvcc
pip install -e .[sglang]
pip install liger-kernel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip uninstall pynvml
pip install "latex2sympy2-extended[antlr4_9_3]"
```

#### Prepare Datasets and Model
1. Download the training dataset **ORZ** from [huggingface](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection) to `experiments/raw_data` folder.
2. `python examples/data_preprocess/orz.py`
3. `python examples/data_preprocess/aime25.py`
4. Download the model **Qwen3-4B-Instruct-2507** from [huggingface](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

#### Training Scripts
Modify the `RAY_DATA_HOME` and `MODEL_PATH` to yours.
```
bash experiments/run.sh
```

### Stage 2: NPR-Beta

#### How to install
```
cd NPR_Beta
conda create -n warmup python=3.11 -y
conda activate warmup
pip install -r requirements.txt
```

#### Prepare Datasets and Model
```
bash scripts/sampling.sh
```
Key parameters in `sampling.sh`:
- `MODEL_PATH`: Path to model checkpoint (Stage 1)
- `OUTPUT_DIR`: Output directory for sampled trajectories
- `--dataset`: Dataset name (default: ORZ-MATH-57K)
- `--instruction`: Prompt template file
- `--max_sample_trial`: Max sampling attempts per problem (default: 8)
- `--temperature`: Sampling temperature (default: 1.0)

#### Training Scripts
```
bash train/sft_math.sh
```
Key parameters in `sft_math.sh`:
- `base_model`: Base model to fine-tune (default: Qwen3-4B-Instruct)
- `train_file_path`: Training data directory (default: `dataset/math/rejection_sampling/train`)
- `lr`: Learning rate
- `epochs`: Number of training epochs
- Output checkpoints saved to `ckpts/NPR-Warmup-4B-Inst-{timestamp}/`

### Stage 3: NPR

#### How to install
```
cd NPR_RL
conda create -n rl python=3.11
conda activate rl
conda install nvidia::cuda-nvcc
pip install -e .[sglang]
pip install liger-kernel
pip uninstall pynvml
pip install "latex2sympy2-extended[antlr4_9_3]"
cd verl/workers/rollout/sglang_rollout/sglang/python
pip install -e .
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
pip install fire
```

#### Prepare Datasets and Model
1. Download the training dataset **ORZ** from [huggingface](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection) to `experiments/raw_data` folder.
2. `python examples/data_preprocess/orz.py` 
3. `python examples/data_preprocess/aime.py` 

#### Training Scripts
Modify the `RAY_DATA_HOME` and `MODEL_PATH` to yours.

Note the `MODEL_PATH` is from Stage 2.
```
bash experiments/run.sh
```

### Evaluation

#### How to install
```
cd evaluation
conda create -n eval python=3.10
conda activate eval
pip install -r requirements.txt
```

#### Prepare Datasets and Model
xxxx

#### Scripts
Modify the `<Model-Path>` to yours.

Note the `<Model-Path>` is from Stage 3 or download from Huggingface. 
```
./scripts/eval.sh \
    --cuda 0,1,2,3,4,5,6,7 \
    --tp_size 2 \
    --dp_size 4 \
    --task "AIME25" \
    --max_eval_samples 30 \
    --eval_batch_size 8 \
    --model_path <Model-Path> \
    --prompt_path prompts/npr.txt \
    --engine parallel \
    --num_samples 1 \
    --k 1 \
    --max_new_tokens 40000 \
    --temperature 1.0 \
    --top_p 0.7 \
    --top_k -1 \
    --overwrite \
    --apply_chat
```

## Citation
```
xx
```

## Acknowledgment
This codebase is influenced by remarkable projects from the LLM community, including [verl](https://github.com/volcengine/verl?tab=readme-ov-file), [MultiVerse](https://github.com/Infini-AI-Lab/Multiverse) and [sglang](https://github.com/sgl-project/sglang).
