"""
Reranker module for improving retrieval quality.

Uses cross-encoder models to rerank retrieved documents based on query relevance.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Global cache for reranker model to avoid reloading on every query
_reranker_cache: dict = {}


class BGEReranker:
    """
    Reranker using BAAI/bge-reranker-base model.
    
    This cross-encoder model scores (query, document) pairs and provides
    more accurate relevance scores than embedding similarity.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        """
        Initialize the reranker.
        
        Args:
            model_name: HuggingFace model name for the cross-encoder
        """
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading reranker model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("Reranker model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
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
            logger.warning("Reranker model not loaded, returning original order")
            return documents
        
        # Create (query, document_text) pairs for scoring
        pairs = [(query, doc.get("text", "")) for doc in documents]
        
        try:
            # Get relevance scores from cross-encoder
            scores = self.model.predict(pairs)
            
            # Attach scores to documents
            for doc, score in zip(documents, scores):
                doc[score_field] = float(score)
            
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


def get_reranker(model_name: str = "BAAI/bge-reranker-base") -> BGEReranker:
    """
    Get a cached reranker instance.
    
    Uses a global cache to avoid reloading the model on every call.
    
    Args:
        model_name: HuggingFace model name
    
    Returns:
        BGEReranker instance
    """
    global _reranker_cache
    
    if model_name not in _reranker_cache:
        _reranker_cache[model_name] = BGEReranker(model_name)
    
    return _reranker_cache[model_name]
