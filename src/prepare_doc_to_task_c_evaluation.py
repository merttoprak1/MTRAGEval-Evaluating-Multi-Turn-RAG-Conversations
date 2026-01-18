import json
from copy import deepcopy

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def deep_merge(base, incoming):
    """
    base: item2 (öncelikli)
    incoming: item1 (eksik olanları tamamlayan)
    """
    print(base, incoming)
    for k, v in incoming.items():
        print(k)
        if k not in base:
            base[k] = deepcopy(v)
    return base

# dosya yolları
file1 = "human/evaluations/reference.jsonl"
file2 = "predictions/task_c/generation/first_100_task_c_qr_rr_qdrant_all-minilm-l6-v2_k20_ms-marco-MiniLM-L-12-v2_k10_openai-gpt-oss-20b_20260118_093622.jsonl"
output_file = "output.jsonl"

data1 = load_jsonl(file1)
data2 = load_jsonl(file2)

map1 = {item["task_id"]: item for item in data1}
map2 = {item["task_id"]: item for item in data2}
print(map1["5b2404d71f9ff7edabddb3b1a8b329e7::4"])
merged = []

for task_id, item2 in map2.items():
    merged_item = deepcopy(item2)
    if task_id in map1:
        merged_item = deep_merge(merged_item, map1[task_id])
    merged.append(merged_item)

save_jsonl(output_file, merged)
