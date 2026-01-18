import json
import argparse
from copy import deepcopy
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def save_jsonl(path, data):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

def deep_merge(base, incoming):
    """
    Merges incoming dictionary into base dictionary.
    Only adds keys from incoming that are NOT present in base.
    
    base: The primary dictionary (predictions)
    incoming: The secondary dictionary to fill gaps (reference)
    """
    for k, v in incoming.items():
        if k not in base:
            base[k] = deepcopy(v)
    return base

def merge_predictions_with_reference(reference_path, prediction_path, output_path):
    """
    Merges validation/reference data structure into predictions file.
    Matches items by 'task_id'.
    """
    logger.info(f"Loading reference data from: {reference_path}")
    data_ref = load_jsonl(reference_path)
    
    logger.info(f"Loading predictions from: {prediction_path}")
    data_pred = load_jsonl(prediction_path)

    # index reference data by task_id
    map_ref = {item["task_id"]: item for item in data_ref}
    
    merged = []
    logger.info("Merging data...")
    
    matched_count = 0
    for item_pred in data_pred:
        task_id = item_pred.get("task_id")
        merged_item = deepcopy(item_pred)
        
        found_ref = None
        if task_id:
            if task_id in map_ref:
                found_ref = map_ref[task_id]
            else:
                # Try handling separator mismatch :: vs <::>
                alt_id_1 = task_id.replace("::", "<::>")
                alt_id_2 = task_id.replace("<::>", "::")
                
                if alt_id_1 in map_ref:
                    found_ref = map_ref[alt_id_1]
                elif alt_id_2 in map_ref:
                    found_ref = map_ref[alt_id_2]
        
        if found_ref:
            merged_item = deep_merge(merged_item, found_ref)
            matched_count += 1
        
        merged.append(merged_item)

    logger.info(f"Merged {len(merged)} items. Matched {matched_count} task_ids from reference.")
    
    logger.info(f"Saving output to: {output_path}")
    save_jsonl(output_path, merged)
    logger.info("Done.")

def main():
    parser = argparse.ArgumentParser(description="Merge prediction JSONL with reference JSONL for Task C evaluation.")
    
    parser.add_argument("--reference", type=str, 
                        default="human/evaluations/reference.jsonl",
                        help="Path to the reference/gold standard JSONL file")
    
    parser.add_argument("--prediction", type=str, required=True,
                        help="Path to the prediction JSONL file")
    
    parser.add_argument("--output", type=str, default="output.jsonl",
                        help="Path to save the merged output JSONL file")

    args = parser.parse_args()

    merge_predictions_with_reference(args.reference, args.prediction, args.output)

if __name__ == "__main__":
    main()
