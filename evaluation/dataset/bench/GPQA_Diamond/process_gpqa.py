import json
import re

def process_gpqa_sample(sample):
    question_text = sample['question']
    
    option_pattern = re.compile(r"([A-Z])\.\s*(.*)")
    options = []
    option_letters = []
    for line in question_text.splitlines():
        m = option_pattern.match(line.strip())
        if m:
            option_letters.append(m.group(1))
            options.append(m.group(2).strip())

    answer_index = option_letters.index(sample['answer'])
    answer_content = options[answer_index]
    
    sample['options'] = options
    sample['answer'] = [sample['answer'], answer_content]
    
    return sample

with open('./gpqa_diamond.json', 'r') as f:
    data = json.load(f)

new_data = [process_gpqa_sample(sample) for sample in data]

with open("./gpqa_diamond_processed.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=2)