import requests
import logging
from typing import List
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

class LocalOllamaEmbeddings(Embeddings):
    """
    Custom Embedding class for local Ollama service.
    Matches the user's specified API structure:
    POST /api/embed
    {
        "model": "model_name",
        "input": ["text1", "text2"]
    }
    """
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip('/')
        self.model = model

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        batch_size = 32
        embeddings = []
        total_texts = len(texts)
        
        logger.info(f"Starting embedding generation for {total_texts} texts with batch_size={batch_size}")
        
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            url = f"{self.base_url}/api/embed"
            payload = {
                "model": self.model,
                "input": batch
            }
            
            if (i // batch_size) % 10 == 0:
                 logger.info(f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")
            
            try:
                response = requests.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                
                if "embeddings" in data:
                    embeddings.extend(data["embeddings"])
                else:
                    logger.error(f"Unexpected response format: {data}")
                    raise ValueError(f"Unexpected response format: {data}")
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch starting at index {i}: {e}", exc_info=True)
                raise

        logger.info(f"Successfully generated {len(embeddings)} embeddings")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        # Reuse embed_documents for single query
        result = self.embed_documents([text])
        if result:
            return result[0]
        return []
