"""
Reranker module for improving retrieval quality.

Uses FlashRank for fast, lightweight reranking of retrieved documents.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global cache for reranker model to avoid reloading on every query
_reranker_cache: dict = {}


class FlashRanker:
    """
    Reranker using FlashRank library.
    
    FlashRank is a lightweight, fast reranking library that uses ONNX
    for efficient CPU inference without requiring PyTorch.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
        """
        Initialize the reranker.
        
        Args:
            model_name: FlashRank model name. Options include:
                - "ms-marco-MiniLM-L-12-v2" (default, good balance)
                - "ms-marco-TinyBERT-L-2-v2" (faster, smaller)
                - "rank-T5-flan" (larger, higher quality)
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the FlashRank model."""
        try:
            from flashrank import Ranker
            
            logger.info(f"Loading FlashRank model: {self.model_name}")
            self.model = Ranker(model_name=self.model_name)
            logger.info("FlashRank model loaded successfully")
            
        except ImportError:
            logger.error("flashrank not installed. Run: pip install flashrank")
            raise ImportError("Please install flashrank: pip install flashrank")
        except Exception as e:
            logger.error(f"Failed to load FlashRank model: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        documents: list[dict], 
        top_k: Optional[int] = None,
        score_field: str = "rerank_score"
    ) -> list[dict]:
        """
        Rerank documents based on query relevance.
        
        Args:
            query: The search query
            documents: List of dicts, each must have a 'text' field
            top_k: Number of top documents to return (None = return all, reordered)
            score_field: Field name to store the rerank score
        
        Returns:
            List of documents sorted by relevance, with rerank scores added
        """
        if not documents:
            return documents
        
        if self.model is None:
            logger.warning("FlashRank model not loaded, returning original order")
            return documents
        
        try:
            from flashrank import RerankRequest
            
            # Create passages for FlashRank
            passages = [
                {"id": i, "text": doc.get("text", "")}
                for i, doc in enumerate(documents)
            ]
            
            # Create rerank request
            rerank_request = RerankRequest(query=query, passages=passages)
            
            # Get reranked results
            results = self.model.rerank(rerank_request)
            
            # Create a mapping from id to rerank score
            score_map = {r["id"]: r["score"] for r in results}
            
            # Attach scores to original documents
            for i, doc in enumerate(documents):
                doc[score_field] = float(score_map.get(i, 0.0))
            
            # Sort by rerank score (descending - higher is more relevant)
            reranked = sorted(documents, key=lambda x: x.get(score_field, 0), reverse=True)
            
            # Return top_k if specified
            if top_k is not None and top_k > 0:
                return reranked[:top_k]
            
            return reranked
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # Return original order on failure
            return documents


def get_reranker(model_name: str = "ms-marco-MiniLM-L-12-v2") -> FlashRanker:
    """
    Get a cached reranker instance.
    
    Uses a global cache to avoid reloading the model on every call.
    
    Args:
        model_name: FlashRank model name
    
    Returns:
        FlashRanker instance
    """
    global _reranker_cache
    
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = FlashRanker(model_name)
    
    return _reranker_cache[model_name]
