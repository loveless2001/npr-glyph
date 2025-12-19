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

    with open("experiments/raw_data/orz_math_57k_collection/orz_math_57k_collection.json", "r", encoding="utf-8") as file:
        dataset = json.load(file)

    processed_data = []
    for item in tqdm(dataset):
        processed_data.append(
            {"input": item[0]['value'].strip(),
             "ground_truth": item[1]['ground_truth']['value'].strip()}
        )
    print(processed_data[0])

    # follow R1
    system_prompt = r"""
You must write your answer **strictly following the glyph-based structure defined below**.
Failure to comply with this structure will result in an **invalid response**.

---

### **Definitions and Rules**

**Structural Glyphs**

* **🜞 CONTEXT-LOCK**
  Sets the objective and context for the current stage of reasoning.

* **🜆i PLAN**
  A single, specific, and actionable task or hypothesis to be executed.
  Multiple 🜆 within the same 🜞 represent **parallel exploration**.

* **🜂i STEP**
  The detailed execution of the corresponding 🜆i plan.
  The number of 🜂 blocks must **exactly match** the number of 🜆 plans in the preceding 🜞.
  **Crucially:** the content of 🜂i must be generated *as if you have no knowledge of the content of its sibling steps*.

* **🜃 TAKEAWAY**
  Analyze the executed 🜂 steps and generate a **concise synthesis**.
  Compare outcomes, identify the most promising path, or consolidate findings.
  **No new computation or reasoning is allowed here** — only analysis of the executed steps.

* **🝞 FINAL**
  The final, user-facing conclusion summarizing the entire logical journey.
  For questions with a definitive short answer, include `\\boxed{...}` containing **only** the final result.

---

### **Strict Requirements**

1. **Execute Independently**

   * For each 🜆 plan, generate a corresponding 🜂 step.
   * Each 🜂 step must be a **self-contained and complete strategy**, based **only** on its associated 🜆 plan.
   * The number of 🜂 steps must **always equal** the number of 🜆 plans in the directly preceding 🜞.
   * Avoid words implying sequence or dependency (e.g. *then*, *after*, *next*).

2. **Explore in Parallel**

   * When a problem involves multiple hypotheses, alternative methods, or independent sub-tasks, the next 🜞 should contain **multiple 🜆 plans**.
   * Each 🜆 represents a **parallel line of reasoning**.
   * A 🜞 with a single 🜆 is allowed if only one plan is needed.
   * Multiple alternative plans are recommended and rewarded.

3. **Meaningful Content Only**

   * All glyph blocks must contain meaningful content.
   * Do **not** add any text outside glyph-structured blocks.

4. **Strict Output Discipline**

   * Do **not** include explanations, commentary, or formatting outside the glyph structure.
   * Do **not** wrap the output in code blocks.
   * Output must consist **only** of glyph-structured content.

---

### **Example Output Format**

```
🜞
🜆1: [A concise one-sentence, independent high-level plan.]
🜆2: [A concise one-sentence, independent high-level plan.]
🜂1: [Detailed analysis trajectory of plan 1. Must be entirely self-contained.]
🜂2: [Detailed analysis trajectory of plan 2. Must be entirely self-contained.]
🜃
[Compare the results from the steps above. Synthesize the findings and determine the next action.]

🜞
🜆1: [A one-sentence, high-level strategy.]
🜂1: [Detailed analysis trajectory of plan 1. Must be entirely self-contained.]
🜃
[Synthesize findings and determine next action.]

🝞
[Final summarized conclusion based on all takeaways. Include definitive answers in \\boxed{...}.]
```
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
                    {"role": "user", "content": question + system_prompt}
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

    dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
