import json
import pandas as pd
from copy import deepcopy

df = pd.read_parquet("./data/test-00000-of-00001.parquet")

json_list = []

for _, row in df.iterrows():
    options = row['options'].tolist()

    question_text = row['question'].strip() + "\n\n"
    for idx, opt in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", options):
        question_text += f"{idx}. {opt}\n"
    question_text = question_text.strip()
    answer_content = options[row['answer_index']]
    
    json_item = {
        "question": question_text,
        "options": options,
        "answer": [row['answer'], answer_content],
    }
    
    json_list.append(json_item)

with open("./mmlu_pro_test_processed.json", "w", encoding="utf-8") as f:
    json.dump(json_list, f, ensure_ascii=False, indent=4)