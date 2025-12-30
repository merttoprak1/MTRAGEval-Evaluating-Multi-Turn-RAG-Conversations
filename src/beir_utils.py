"""
BEIR Format Utilities for Task A Retrieval Evaluation.

This module provides functions to load and parse BEIR format files:
- Qrels (relevance judgments): TSV format with query-id, corpus-id, score
- Queries: JSONL format with _id and text fields
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

# Base paths for retrieval tasks
RETRIEVAL_TASKS_BASE = Path(__file__).parent.parent.parent / "human" / "retrieval_tasks"

# Available corpora
AVAILABLE_CORPORA = ["clapnq", "cloud", "fiqa", "govt"]

# Query types
QUERY_TYPES = {
    "questions": "All Questions (Full Conversation)",
    "lastturn": "Last Turn Only", 
    "rewrite": "Rewritten Queries"
}


def get_retrieval_task_paths(corpus: str, query_type: str = "questions") -> Dict[str, Path]:
    """
    Get file paths for a given corpus and query type.
    
    Args:
        corpus: One of clapnq, cloud, fiqa, govt
        query_type: One of questions, lastturn, rewrite
        
    Returns:
        Dictionary with 'queries' and 'qrels' paths
    """
    if corpus not in AVAILABLE_CORPORA:
        raise ValueError(f"Unknown corpus: {corpus}. Available: {AVAILABLE_CORPORA}")
    
    corpus_dir = RETRIEVAL_TASKS_BASE / corpus
    
    # Map query type to filename
    query_file_map = {
        "questions": f"{corpus}_questions.jsonl",
        "lastturn": f"{corpus}_lastturn.jsonl",
        "rewrite": f"{corpus}_rewrite.jsonl"
    }
    
    query_file = query_file_map.get(query_type, f"{corpus}_questions.jsonl")
    
    return {
        "queries": corpus_dir / query_file,
        "qrels": corpus_dir / "qrels" / "dev.tsv"
    }


def load_qrels(qrels_path: Path) -> Dict[str, Dict[str, int]]:
    """
    Load qrels (relevance judgments) from TSV file.
    
    BEIR qrels format:
    query-id    corpus-id    score
    
    Args:
        qrels_path: Path to dev.tsv file
        
    Returns:
        Nested dict: {query_id: {doc_id: relevance_score}}
    """
    qrels = {}
    
    if not qrels_path.exists():
        logger.warning(f"Qrels file not found: {qrels_path}")
        return qrels
    
    with open(qrels_path, 'r', encoding='utf-8') as f:
        # Skip header
        header = f.readline()
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], int(parts[2])
                
                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score
    
    logger.info(f"Loaded {len(qrels)} queries with relevance judgments from {qrels_path}")
    return qrels


def load_queries(queries_path: Path) -> Dict[str, str]:
    """
    Load queries from JSONL file.
    
    BEIR queries format (JSONL):
    {"_id": "query_id", "text": "query text..."}
    
    Args:
        queries_path: Path to *_questions.jsonl file
        
    Returns:
        Dict: {query_id: query_text}
    """
    queries = {}
    
    if not queries_path.exists():
        logger.warning(f"Queries file not found: {queries_path}")
        return queries
    
    with open(queries_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            query_id = data.get("_id", "")
            query_text = data.get("text", "")
            if query_id and query_text:
                queries[query_id] = query_text
    
    logger.info(f"Loaded {len(queries)} queries from {queries_path}")
    return queries


def load_corpus(corpus_name: str) -> Dict[str, Dict[str, str]]:
    """
    Load corpus documents from passage_level JSONL.
    
    Args:
        corpus_name: One of clapnq, cloud, fiqa, govt
        
    Returns:
        Dict: {doc_id: {"text": ..., "title": ...}}
    """
    corpus_path = Path(__file__).parent.parent.parent / "corpora" / "passage_level" / f"{corpus_name}.jsonl"
    
    corpus = {}
    
    if not corpus_path.exists():
        logger.warning(f"Corpus file not found: {corpus_path}")
        return corpus
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            doc_id = data.get("_id", data.get("id", ""))
            if doc_id:
                corpus[doc_id] = {
                    "text": data.get("text", ""),
                    "title": data.get("title", "")
                }
    
    logger.info(f"Loaded {len(corpus)} documents from {corpus_path}")
    return corpus


def calculate_retrieval_metrics(
    retrieved_results: Dict[str, List[str]], 
    qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [1, 3, 5, 10]
) -> Dict[str, Dict[str, float]]:
    """
    Calculate retrieval metrics: Recall@K, nDCG@K, MRR.
    
    Args:
        retrieved_results: {query_id: [doc_id1, doc_id2, ...]} ordered by score
        qrels: {query_id: {doc_id: relevance}} ground truth
        k_values: List of K values for metrics
        
    Returns:
        Dict with metrics for each K value
    """
    import math
    
    metrics = {f"Recall@{k}": [] for k in k_values}
    metrics.update({f"nDCG@{k}": [] for k in k_values})
    metrics["MRR"] = []
    
    for query_id, retrieved_docs in retrieved_results.items():
        if query_id not in qrels:
            continue
            
        relevant_docs = set(qrels[query_id].keys())
        
        # MRR - Mean Reciprocal Rank
        rr = 0.0
        for i, doc_id in enumerate(retrieved_docs):
            if doc_id in relevant_docs:
                rr = 1.0 / (i + 1)
                break
        metrics["MRR"].append(rr)
        
        # Recall@K and nDCG@K
        for k in k_values:
            top_k = retrieved_docs[:k]
            
            # Recall@K
            hits = len(set(top_k) & relevant_docs)
            recall = hits / len(relevant_docs) if relevant_docs else 0
            metrics[f"Recall@{k}"].append(recall)
            
            # nDCG@K
            dcg = 0.0
            for i, doc_id in enumerate(top_k):
                if doc_id in relevant_docs:
                    rel = qrels[query_id].get(doc_id, 0)
                    dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0
            
            # Ideal DCG
            ideal_rels = sorted([qrels[query_id].get(d, 0) for d in relevant_docs], reverse=True)[:k]
            idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))
            
            ndcg = dcg / idcg if idcg > 0 else 0
            metrics[f"nDCG@{k}"].append(ndcg)
    
    # Average all metrics
    avg_metrics = {}
    for metric_name, values in metrics.items():
        avg_metrics[metric_name] = sum(values) / len(values) if values else 0.0
    
    return avg_metrics


def format_beir_results_for_eval(
    query_id: str,
    retrieved_docs: List[Tuple[str, float]],  # [(doc_id, score), ...]
    corpus: Dict[str, Dict[str, str]]
) -> Dict:
    """
    Format retrieval results for evaluation script.
    
    Args:
        query_id: Query identifier
        retrieved_docs: List of (doc_id, score) tuples
        corpus: Loaded corpus documents
        
    Returns:
        Dict in MTRAG format for evaluation
    """
    contexts = []
    for doc_id, score in retrieved_docs:
        doc = corpus.get(doc_id, {})
        contexts.append({
            "document_id": doc_id,
            "text": doc.get("text", ""),
            "title": doc.get("title", ""),
            "score": score
        })
    
    return {
        "task_id": query_id,
        "contexts": contexts
    }
