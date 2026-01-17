"""
Utility functions for generating prediction filenames.

This module provides functions to generate structured filenames for prediction files
based on task configuration parameters like vector DB, embedding model, reranker settings, etc.
"""


def generate_prediction_filename(
    task_type: str,
    timestamp: str,
    vector_db_type: str = None,
    embedding_model: str = None,
    top_k: int = None,
    has_reranker: bool = False,
    reranker_model: str = None,
    top_k_reranker: int = None,
    global_llm_name: str = None,
    query_rewritten: bool = False
) -> str:
    """
    Generate prediction filename based on task configuration.
    
    Args:
        task_type: Task identifier (e.g., 'task_a', 'task_b', 'task_c')
        timestamp: Timestamp string
        vector_db_type: Vector database type (e.g., 'qdrant', 'faiss')
        embedding_model: Embedding model name
        top_k: Number of top documents retrieved
        has_reranker: Whether reranker is enabled
        reranker_model: Reranker model name
        top_k_reranker: Number of top reranked documents
        global_llm_name: LLM model name
        query_rewritten: Whether query rewrite is enabled
    
    Returns:
        str: Generated filename without extension
    """
    parts = [task_type]
    
    # Add query rewrite flag right after task name
    if query_rewritten:
        parts.append("qr")
    
    # Add reranker flag after qr
    if has_reranker:
        parts.append("rr")
    
    # Add retrieval-related components (for Task A and C)
    if vector_db_type:
        parts.append(vector_db_type)
    
    if embedding_model:
        # Clean embedding model name (replace special chars)
        clean_embed = embedding_model.replace("/", "-").replace(":", "-")
        parts.append(clean_embed)
    
    if top_k is not None:
        parts.append(f"k{top_k}")
    
    # Add reranker model and top-k if reranker is enabled
    if has_reranker and reranker_model:
        # Clean reranker model name
        clean_reranker = reranker_model.replace("/", "-").replace(":", "-")
        parts.append(clean_reranker)
        
        if top_k_reranker is not None:
            parts.append(f"k{top_k_reranker}")
    
    # Add LLM name
    if global_llm_name:
        # Clean LLM name (replace special chars)
        clean_llm = global_llm_name.replace("/", "-").replace(":", "-")
        parts.append(clean_llm)
    
    # Add timestamp
    parts.append(timestamp)
    
    return "_".join(parts)


def generate_taskc_generation_filename(
    task_type: str,
    timestamp: str,
    vector_db_type: str = None,
    embedding_model: str = None,
    top_k: int = None,
    has_reranker: bool = False,
    reranker_model: str = None,
    top_k_reranker: int = None,
    global_llm_name: str = None,
    query_rewritten: bool = False
) -> str:
    """
    Generate Task C generation filename with shorter notation.
    
    This is an alias for generate_prediction_filename with the same parameters,
    maintained for clarity when generating Task C generation filenames.
    
    Uses 'k' prefix instead of 'top', 'qr' for query rewrite, 'rr' for reranker.
    
    Args:
        task_type: Task identifier (e.g., 'task_c')
        timestamp: Timestamp string
        vector_db_type: Vector database type (e.g., 'qdrant', 'faiss')
        embedding_model: Embedding model name
        top_k: Number of top documents retrieved
        has_reranker: Whether reranker is enabled
        reranker_model: Reranker model name
        top_k_reranker: Number of top reranked documents
        global_llm_name: LLM model name
        query_rewritten: Whether query rewrite is enabled
    
    Returns:
        str: Generated filename without extension
    
    """
    # Use the same logic as generate_prediction_filename
    return generate_prediction_filename(
        task_type=task_type,
        timestamp=timestamp,
        vector_db_type=vector_db_type,
        embedding_model=embedding_model,
        top_k=top_k,
        has_reranker=has_reranker,
        reranker_model=reranker_model,
        top_k_reranker=top_k_reranker,
        global_llm_name=global_llm_name,
        query_rewritten=query_rewritten
    )
