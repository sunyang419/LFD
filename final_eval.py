import json
import os
from collections import OrderedDict
from utils.normalize_answer import normalize_answer


folder = "./output/qwen3-8b/2wiki/"

# read all JSON files
json_files = [f for f in os.listdir(folder) if f.endswith('.json') and os.path.isfile(os.path.join(folder, f)) and not f.startswith("final_eval") and not f.startswith("ef_eval") and not f.startswith("nq_lfd_ef")]
result = {}

for file_name in json_files:
    file_path = os.path.join(folder, file_name)
    prefix = file_name.split('.')[0]
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    right, wrong = 0, 0
    for datum in data:
        output = datum["output"].split('\n')[0]
        answers = datum["answers"]
        flag = False
        for answer in answers:
            if normalize_answer(answer) in normalize_answer(output):
                flag = True
                break
        if flag:
            right += 1
        else:
            wrong += 1
    
    accuracy = right / (wrong + right) if (wrong + right) else 0
    result[prefix] = accuracy

sorted_result = OrderedDict(sorted(result.items(), key=lambda x: x[1], reverse=True))

json.dump(sorted_result, open(f'{folder}final_eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)