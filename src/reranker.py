"""
Reranker module for improving retrieval quality.

Supports multiple reranker backends:
- FlashRank: Lightweight, fast, ONNX-based (no PyTorch required)
- BGE: Higher quality, requires sentence-transformers/PyTorch
"""

import logging
from typing import Optional, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Global cache for reranker models to avoid reloading
_reranker_cache: dict = {}

# Available reranker types
RERANKER_TYPES = {
    "flashrank": "FlashRank (Lightweight, Fast)",
    "bge": "BGE Reranker (Higher Quality, Requires PyTorch)",
    "other": "Other"
}

FLASHRANK_MODELS = {
    "ms-marco-MiniLM-L-12-v2": "MiniLM-L12 (Default, Good Balance)",
    "ms-marco-TinyBERT-L-2-v2": "TinyBERT-L2 (Faster, Smaller)",
    "rank-T5-flan": "T5-Flan (Larger, Higher Quality)"
}

BGE_MODELS = {
    "BAAI/bge-reranker-base": "BGE Base (Default)",
    "BAAI/bge-reranker-large": "BGE Large (Higher Quality)",
    "BAAI/bge-reranker-v2-m3": "BGE v2 M3 (Multilingual)"
}

OTHER_MODELS = {
    "Qwen/Qwen3-Reranker-8B": "Qwen/Qwen3-Reranker-8B",
    "cross-encoder/ms-marco-MiniLM-L6-v2": "cross-encoder/ms-marco-MiniLM-L6-v2",
    "mixedbread-ai/mxbai-rerank-xsmall-v1": "mixedbread-ai/mxbai-rerank-xsmall-v1"
}


class BaseReranker(ABC):
    """Abstract base class for rerankers."""
    
    @abstractmethod
    def rerank(
        self, 
        query: str, 
        documents: list[dict], 
        top_k: Optional[int] = None,
        score_field: str = "rerank_score"
    ) -> list[dict]:
        """Rerank documents based on query relevance."""
        pass


class FlashRanker(BaseReranker):
    """
    Reranker using FlashRank library.
    
    FlashRank is a lightweight, fast reranking library that uses ONNX
    for efficient CPU inference without requiring PyTorch.
    """
    
    def __init__(self, model_name: str = "ms-marco-MiniLM-L-12-v2"):
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
        """Rerank documents based on query relevance."""
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
            logger.error(f"FlashRank reranking failed: {e}")
            return documents


class BGEReranker(BaseReranker):
    """
    Reranker using BGE (BAAI) cross-encoder models.
    
    Uses sentence-transformers CrossEncoder for high-quality reranking.
    Requires PyTorch and sentence-transformers.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the BGE cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder
            
            logger.info(f"Loading BGE model: {self.model_name}")
            self.model = CrossEncoder(self.model_name)
            logger.info("BGE model loaded successfully")
            
        except ImportError:
            logger.error("sentence-transformers not installed. Run: pip install sentence-transformers")
            raise ImportError("Please install sentence-transformers: pip install sentence-transformers")
        except Exception as e:
            logger.error(f"Failed to load BGE model: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        documents: list[dict], 
        top_k: Optional[int] = None,
        score_field: str = "rerank_score"
    ) -> list[dict]:
        """Rerank documents based on query relevance."""
        if not documents:
            return documents
        
        if self.model is None:
            logger.warning("BGE model not loaded, returning original order")
            return documents
        
        try:
            # Create (query, document_text) pairs for scoring
            pairs = [(query, doc.get("text", "")) for doc in documents]
            
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
            logger.error(f"BGE reranking failed: {e}")
            return documents


def get_reranker(
    reranker_type: str = "flashrank",
    model_name: Optional[str] = None
) -> BaseReranker:
    """
    Get a cached reranker instance.
    
    Args:
        reranker_type: "flashrank" or "bge"
        model_name: Optional model name. If None, uses default for the type.
    
    Returns:
        Reranker instance
    """
    global _reranker_cache
    
    # Set default model names
    if model_name is None:
        if reranker_type == "flashrank":
            model_name = "ms-marco-MiniLM-L-12-v2"
        else:
            model_name = "BAAI/bge-reranker-base"
    
    cache_key = f"{reranker_type}:{model_name}"
    
    if cache_key not in _reranker_cache:
        if reranker_type == "flashrank":
            _reranker_cache[cache_key] = FlashRanker(model_name)
        elif reranker_type == "bge":
            _reranker_cache[cache_key] = BGEReranker(model_name)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}. Use 'flashrank' or 'bge'.")
    
    return _reranker_cache[cache_key]
