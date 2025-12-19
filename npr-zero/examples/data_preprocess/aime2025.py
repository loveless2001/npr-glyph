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
Preprocess the GSM8k dataset to parquet format
"""

import argparse
import os

import datasets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="experiments/data/math_total")
    parser.add_argument("--hdfs_dir", default=None)

    args = parser.parse_args()

    data_source = "aime2025"

    dataset = datasets.load_dataset("math-ai/aime25", "default")
    # dataset.save_to_disk("experiments/raw_data/aime2025")

    test_dataset = dataset["test"]

    print(test_dataset)

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
            question = example.pop("problem")
            solution = example.pop("answer")

            data = {
                "data_source": data_source,
                "prompt": [
                    # {"role": "system", "content": system_prompt},
                    {"role": "user", "content": question + system_prompt}
                ],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": solution.strip()},
                "extra_info": {
                    "apply_chat_template": True,
                    "split": split,
                    "index": idx,
                    "answer": solution,
                    "question": question,
                },
            }
            return data

        return process_fn

    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    local_dir = args.local_dir
    print(test_dataset[0])

    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

    # Conversion for rejection_sampling.py (MegaScience format)
    import json
    
    megascience_data = []
    # We can iterate over the test_dataset which has already been mapped
    for item in test_dataset:
        # The 'prompt' field in the processed dataset is a list of dicts: [{"role": "user", "content": "..."}]
        # We need to extract the original question.
        # However, the 'prompt' content includes the system prompt appended.
        # We can use the 'extra_info' field which we preserved!
        
        question = item["extra_info"]["question"]
        solution = item["extra_info"]["answer"]
        
        megascience_data.append({
            "problem": question,
            "solution": solution
        })

    # Save to the specific path expected by the sampling script
    megascience_path = "dataset/megascience/megascience.json"
    os.makedirs(os.path.dirname(megascience_path), exist_ok=True)
    
    with open(megascience_path, "w") as f:
        json.dump(megascience_data, f, indent=4)

    print(f"Also saved JSON for evaluation to {megascience_path}")
