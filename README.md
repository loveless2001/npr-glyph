# Glyph-NPR: Native Parallel Reasoning with Glyphs

This repository is a fork of the original [Native Parallel Reasoning (NPR)](https://github.com/bigai-nlco/Native-Parallel-Reasoner) project, modified to experiment with a **Glyph-based prompts** instead of XML tags.

For the original documentation and theoretical background, please refer to the [original repository](https://github.com/bigai-nlco/Native-Parallel-Reasoner).

## Experiment: Glyph-Based Structure

We replace the verbose XML tags (`<guideline>`, `<plan>`, `<step>`, etc.) with concise, atomic **Glyph tokens**. This aims to improve token efficiency, simplify parsing, and potentially enhance model stability by treating structural markers as single semantic units.

### The Glyph System

| Concept | Glyph | Semantic Init | Description |
| :--- | :---: | :--- | :--- |
| **Guideline** | `🜞` | "guideline" | Starts a high-level reasoning block plan. |
| **Plan** | `🜆` | "plan" | Defines a specific sub-task or parallel branch. |
| **Step** | `🜂` | "step" | Executes a reasoning step within a plan. |
| **Takeaway** | `🜃` | "takeaway" | Synthesizes results from parallel steps. |
| **Final** | `🝞` | "final" | Marks the final answer (replaces `\boxed`). |

**Key Features:**
*   **Atomic Tokens**: Each glyph is added as a special token (IDs ~151675+) to the tokenizer.
*   **Implicit Closing**: Blocks are closed implicitly by the start of a new peer or parent block (e.g., a new `🜂` closes the previous `🜂`), removing the need for `</step>` closing tags.
*   **Semantic Initialization**: Glyph embeddings are initialized from the mean embeddings of their semantic descriptions (e.g., `🜂` is init from "step") to speed up convergence.

### Reproduction Steps (Runpod)

Follow these steps to reproduce the Glyph-NPR experiment.

#### 1. Installation

```bash
cd npr-beta
conda create -n npr_glyph python=3.11 -y
conda activate npr_glyph
pip install -r requirements.txt
```

#### 2. Sampling (Data Generation)

Generate Glyph-formatted training data using a base instruction-tuned model (e.g., Qwen2.5-Math-7B).

1.  Edit `scripts/sampling.sh`:
    *   Set `MODEL_PATH` to your base model (e.g., `Qwen/Qwen2.5-Math-7B-Instruct`).
    *   Ensure the script uses `prompts/npr_glyph.txt`.

2.  Run sampling:
    ```bash
    bash scripts/sampling.sh
    ```
    *Output will be saved to `data/math_sampling_...jsonl`.*

#### 3. Training (SFT)

Fine-tune the model on the generated Glyph sequences.

1.  Edit `train/sft_math.sh`:
    *   Set `MODEL_PATH` to your base model.
    *   Set `train_file_path` to the output JSONL from step 2.
    *   Set `output_dir` for your checkpoints.

2.  Run training:
    ```bash
    bash train/sft_math.sh
    ```

### Verified Components

*   **Tokenizer**: Glyphs are atomic and round-trip safe. `<think>` tags are preserved for compatibility.
*   **Logic**: Attention masks handling implicit closing have been verified.
*   **Rewards**: Format checkers in `npr-beta/utils/grader.py` and `npr-zero/verl/utils/reward_score` now strictly validate the Glyph structure (`🜞` -> `🜆` -> `🜂` -> `🜃` -> `🝞`).
