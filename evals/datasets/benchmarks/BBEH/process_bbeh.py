import re
import json
from copy import deepcopy

def process_normal_sample(sample):
    new_sample = deepcopy(sample)
    new_sample['question'] = new_sample.pop('input')
    new_sample['answer'] = [new_sample.pop('target')]
    return new_sample

def trans_boardgame_qa_to_mcq(sample):
    new_sample = deepcopy(sample)
    new_sample['question'] = new_sample.pop('input') + "\n\nA. proved\nB. disproved\nC.unknown"
    new_sample['options'] = ['proved', 'disproved', 'unknown']
    answer = new_sample.pop('target')
    new_sample['answer'] = [answer, {'proved': 'A', 'disproved': 'B', 'unknown': 'C'}[answer]]
    return new_sample

def trans_causal_understanding_to_mcq(sample):
    new_sample = deepcopy(sample)
    new_sample['question'] = new_sample.pop('input') + "\n\nA. Yes\nB. No\nC.Ambiguous"
    new_sample['options'] = ['Yes', 'No', 'Ambiguous']
    answer = new_sample.pop('target')
    new_sample['answer'] = [answer, {'Yes': 'A', 'No': 'B', 'Ambiguous': 'C'}[answer]]
    return new_sample

def process_mcq(sample):
    new_sample = deepcopy(sample)
    new_sample['question'] = new_sample.pop('input')

    option_pattern = re.compile(r"(\([A-Z]\))\s*(.*)")
    options = []
    option_letters = []
    for line in new_sample['question'].splitlines():
        m = option_pattern.match(line.strip())
        if m:
            option_letters.append(m.group(1))
            options.append(m.group(2).strip())

    answer = new_sample['target']
    if answer[0] != '(':
        answer = answer_alia = f'({answer})'
    else:
        answer_alia = answer.strip()[1]
    answer_index = option_letters.index(answer)
    answer_content = options[answer_index]

    new_sample['options'] = options
    new_sample['answer'] = [new_sample.pop('target'), answer_alia, answer_content]
    return new_sample


MCQ_TASKS = [
    'disambiguation qa',
    'hyperbaton',
    'shuffled objects',
    'geometric shapes',
    'boolean expressions',
    'movie recommendation',
    'nycc',
]

with open('./bbeh.json', 'r') as f:
    data = json.load(f)

new_data = []
for sample in data:
    task = sample['task']
    if task == 'boardgame qa':
        new_sample = trans_boardgame_qa_to_mcq(sample)
    elif task == 'causal understanding':
        new_sample = trans_causal_understanding_to_mcq(sample)
    elif task in MCQ_TASKS:
        new_sample = process_mcq(sample)
    else:
        new_sample = process_normal_sample(sample)
    new_data.append(new_sample)
with open("./bbeh_processed.json", "w", encoding="utf-8") as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)



