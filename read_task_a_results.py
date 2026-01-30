import pandas as pd
import ast
from pathlib import Path


def parse_filename(fname: str):
    fname = fname.lower()

    def yesno(key):
        return "YES" if key in fname else "NO"

    return {
        "QueryRewrite": yesno("_qr_"),
        "RerankUsed": yesno("_rr_"),
        "VectorDB": "qdrant" if "qdrant" in fname else ("faiss" if "faiss" in fname else None),
        "QR LLM Model": (
            "openai/gpt-oss-20b" if "openai-gpt-oss" in fname else
            "google/gemma-3-12b" if "gemma-3-12b" in fname else None
        ),
        "Embedding Model": (
            "all-minilm:l6-v2" if "minilm" in fname else
            "nomic-embed-text" if "nomic" in fname else
            "jina/jina-embeddings-v2-small-en" if "jina-embed" in fname else
            "snowflake-arctic-embed:137m" if "snowflake" in fname else None
        ),
        "Rerank Model": (
            "BGE v2 M3 (Multilingual)" if "bge" in fname else
            "jinaai/jina-reranker-v2-base-multilingual" if "jina-reranker" in fname else
            "mixedbread-ai/mxbai-rerank-xsmall-v1" if "mixed" in fname else
            "cross-encoder/ms-marco-MiniLM-L6-v2" if "cross-encoder" in fname else None
        ),
        "Rerank Documents": (
            "10" if "k10" in fname else
            "5" if "k5" in fname else None
        )
    }


def extract_scores(csv_path: Path):
    df = pd.read_csv(csv_path)

    last_row = df.iloc[-1]
    cell_3 = ast.literal_eval(str(last_row.iloc[2]))
    cell_4 = ast.literal_eval(str(last_row.iloc[3]))

    scores = list(cell_3) + list(cell_4)
    return {f"Score_{i+1}": scores[i] for i in range(8)}


def build_score_table_from_folder(folder_path: str, out_csv: str):
    rows = []
    for csv_file in Path(folder_path).glob("*.csv"):
        meta = parse_filename(csv_file.name)
        scores = extract_scores(csv_file)
        rows.append({**meta, **scores})

    result_df = pd.DataFrame(rows)
    result_df.to_csv(out_csv, index=False)
    return result_df


# Example usage:
build_score_table_from_folder("data/evaluation/task_a/", "final_score_table.csv")
