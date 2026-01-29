import json

input_jsonl = "data/input_dev/input_task_a_c.jsonl"
context_jsonl = "data/predictions/task_c/retrieval/task_c_qr_rr_qdrant_all-minilm-l6-v2_k20_BAAI-bge-reranker-v2-m3_k5_google-gemma-3-12b_20260128_191307.jsonl"
output_jsonl = "data/input_dev/input_task_a_c_w_query_rewrited_w_gemma.jsonl"

# Read rewritten_query from second file (matched by task_id)
rewritten_map = {}

with open(context_jsonl, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        cid = obj.get("task_id")
        rq = obj.get("rewritten_query")
        if cid and rq:
            rewritten_map[cid] = rq

# Process first file and replace last user input
line_counter = 0
with open(input_jsonl, "r", encoding="utf-8") as fin, \
     open(output_jsonl, "w", encoding="utf-8") as fout:

    for line in fin:
        obj = json.loads(line)
        cid = obj.get("task_id")

        if cid in rewritten_map and "input" in obj:
            # Find last user message
            for msg in reversed(obj["input"]):
                if msg.get("speaker") == "user":
                    msg["text"] = rewritten_map[cid]
                    line_counter += 1
                    break

        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
print(line_counter)
print("Done. Output written to:", output_jsonl)
