# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the Law dataset to parquet format
"""

import json
import os
import datasets

import argparse
from tqdm import tqdm


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='experiments/data/math_total')
    parser.add_argument('--chat_model', action='store_false', help='Whether to use chat model')

    args = parser.parse_args()

    data_source = 'math_dapo'

    dataset = datasets.load_from_disk("experiments/raw_data/dapo_math")["train"]

    processed_data = []
    for item in tqdm(dataset):
        processed_data.append(
            {"input": item['prompt'].strip(),
             "ground_truth": item['solution'].strip()}
        )
    print(processed_data[0])

    # Updated to Glyph-based NPR prompt
    system_prompt = r"""
You are a Native Parallel Reasoner (NPR), an AI capable of simultaneous multi-path reasoning to solve complex problems.
You must use the following Glyph-based structure for your reasoning. The special glyphs `🜞`, `🜆`, `🜂`, `🜃`, and `🝞` define the reasoning flow.

**Structure & Rules:**

1.  **Guideline (🜞)**:
    *   Start a reasoning block with `🜞`.
    *   State the objective or high-level direction for this block.
    *   Immediately follow with one or more **Plans (🜆)**.
    *   Format: `🜞 [Objective] 🜆 [Plan 1] 🜆 [Plan 2] ...`

2.  **Steps (🜂)**:
    *   For *each* Plan (🜆), you must execute a corresponding **Step (🜂)**.
    *   The number of Steps must exactly match the number of Plans in the preceding Guideline.
    *   Steps are executed in parallel: The content of `🜂 Step 1` must NOT depend on `🜂 Step 2` or vice-versa. They share the same context (Guideline).
    *   Format:
        ```
        🜂 [Execution of Plan 1...]
        🜂 [Execution of Plan 2...]
        ...
        ```

3.  **Takeaway (🜃)**:
    *   After all Steps are complete, generate a **Takeaway (🜃)** to synthesize the results.
    *   Analyze the outcomes of parallel steps, resolve conflicts, or consolidate findings.
    *   Decide the next move: either start a new Guideline (🜞) or conclude with the Final Answer.

4.  **Final Answer (🝞)**:
    *   When the solution is found, use the **Final Answer (🝞)** glyph.
    *   State the conclusion clearly.
    *   For math/objective problems, include the final result in `\boxed{...}` *inside* the completion.

**Example Format:**

🜞 We need to simplify the expression. 🜆 Expand terms. 🜆 Factorize.
🜂 [Details of expansion...]
🜂 [Details of factorization...]
🜃 Both methods give result X. Factorization was faster.
🜞 Check boundary conditions. 🜆 Case x=0.
🜂 [Analyzing x=0...]
🜃 Valid for x=0.
🝞 The final answer is \boxed{X}.

**Strict Requirements:**
*   Use the exact glyphs.
*   Maintain the Plan-Step correspondence (N Plans -> N Steps).
*   Keep Steps independent within a block.
*   Always conclude with 🝞 and `\boxed{}` for answers.
"""
    print(system_prompt)

    # add a row to each data item that represents a unique id
    def make_map_fn(split):

        def process_fn(example, idx):

            question = example.pop("input")
            answer = example.pop("ground_truth")

            data = {
                "data_source": data_source,
                "prompt": [
                    # {"role": "system", "content": system_prompt}, 
                    {"role": "user", "content": question + "\n" + system_prompt}
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    'apply_chat_template': True,
                    "split": split,
                    'index': idx,
                    "answer": answer,
                    "question": question,
                }
            }
            return data

        return process_fn
    

    dataset = datasets.Dataset.from_list(processed_data)

    dataset = dataset.map(function=make_map_fn("train"), with_indices=True)

    print(dataset[0])

    local_dir = args.local_dir

    dataset.to_parquet(os.path.join(local_dir, 'train_dapo.parquet'))
