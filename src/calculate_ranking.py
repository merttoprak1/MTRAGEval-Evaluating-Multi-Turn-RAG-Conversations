import json
import statistics

def append_ranking_score(file_path):
    """
    Reads the results file, calculates the MTRAG ranking score (Harmonic Mean),
    and appends it as a summary line at the end of the file.
    """
    scores_rl_f = []
    scores_rb_llm = []
    scores_rb_alg = []

    # 1. Read existing metrics from the file
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        try:
            data = json.loads(line)
            metrics = data.get("metrics", {})
            # Use the IDK-conditioned keys per the paper instructions
            scores_rl_f.append(metrics.get("RL_F_idk", [0.0])[0] or 0.0)
            scores_rb_llm.append(metrics.get("RB_llm_idk", [0.0])[0] or 0.0)
            scores_rb_alg.append(metrics.get("RB_agg_idk", [0.0])[0] or 0.0)
        except:
            continue

    # 2. Calculate Harmonic Mean
    # Formula: 3 / (1/RL + 1/LLM + 1/Alg)
    if not scores_rl_f:
        ranking_score = 0.0
    else:
        avg_rl = statistics.mean(scores_rl_f)
        avg_llm = statistics.mean(scores_rb_llm)
        avg_alg = statistics.mean(scores_rb_alg)
        
        # Avoid division by zero
        if 0 in [avg_rl, avg_llm, avg_alg]:
            ranking_score = 0.0
        else:
            ranking_score = 3 / ((1/avg_rl) + (1/avg_llm) + (1/avg_alg))

    # 3. Append the score to the file
    summary_data = {
        "type": "dataset_summary",
        "ranking_score": ranking_score,
        "breakdown": {
            "avg_RL_F_idk": avg_rl if scores_rl_f else 0,
            "avg_RB_llm_idk": avg_llm if scores_rl_f else 0,
            "avg_RB_agg_idk": avg_alg if scores_rl_f else 0
        }
    }
    
    # Append to file
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write('\n' + json.dumps(summary_data))
        
    return ranking_score