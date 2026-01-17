
import logging
import os
from tempfile import TemporaryDirectory

from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

from src.file_manager import FileManager
from src.embeddings import LocalOllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.config import Config

logger = logging.getLogger(__name__)

# Supported Vector DB types
SUPPORTED_VECTOR_DBS = ["FAISS", "Qdrant"]

def _get_embedding_model(embedding_config: dict):
    """
    Factory function to create embedding model based on config.
    """
    if not embedding_config:
        logger.warning("No embedding config provided, falling back to OpenAI default")
        return OpenAIEmbeddings()
    
    provider = embedding_config.get("provider", "OpenAI")
    logger.info(f"Using embedding provider: {provider}")
    
    if provider == "OpenAI":
        api_key = embedding_config.get("api_key")
        model_name = embedding_config.get("model_name", "text-embedding-3-small")
        if not api_key:
            logger.error("API Key missing for OpenAI embeddings")
            raise ValueError("API Key is required for OpenAI embeddings.")
        logger.info(f"Using OpenAI embedding model: {model_name}")
        return OpenAIEmbeddings(api_key=api_key, model=model_name)
        
    elif provider == "Gemini":
        api_key = embedding_config.get("api_key")
        if not api_key:
            logger.error("API Key missing for Gemini embeddings")
            raise ValueError("API Key is required for Gemini embeddings.")
        model_name = embedding_config.get("model_name", "models/embedding-001")
        return GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model_name)
        
    elif provider == "Local":
        base_url = embedding_config.get("base_url")
        model_name = embedding_config.get("model_name")
        logger.info(f"Configuring Local embeddings: URL={base_url}, Model={model_name}")
        if not base_url or not model_name:
            logger.error("Base URL or Model Name missing for Local embeddings")
            raise ValueError("Base URL and Model Name are required for Local embeddings.")
        return LocalOllamaEmbeddings(base_url=base_url, model=model_name)
    
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

def _load_faiss(
    embedding_model,
    collection_name: str,
    embedding_config: dict = None
):
    """
    Loads an existing FAISS collection.
    """
    model_name = embedding_config.get("model_name", "default_model") if embedding_config else "default_model"
    persist_directory = FileManager.get_collection_path("faiss", model_name, collection_name)
    
    index_faiss_path = os.path.join(persist_directory, "index.faiss")
    index_pkl_path = os.path.join(persist_directory, "index.pkl")

    if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
        logger.info(f"Loading existing FAISS collection: {collection_name} from {persist_directory}")
        try:
            vector_store = FAISS.load_local(
                persist_directory,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded {vector_store.index.ntotal} existing vectors")
            return vector_store
        except Exception as e:
            logger.error(f"Failed to load existing FAISS index: {e}")
            raise
    else:
        logger.warning(f"FAISS collection '{collection_name}' not found at {persist_directory}")
        return None

def _load_qdrant(
    embedding_model,
    collection_name: str,
    db_config: dict = None,
    embedding_config: dict = None
):
    """
    Connects to an existing Qdrant collection.
    """
    db_config = db_config or {}
    qdrant_config = db_config.get("collection", db_config) # Handle potential nested config structure
    
    url = qdrant_config.get("url") or Config.QDRANT_URL
    api_key = qdrant_config.get("api_key") or Config.QDRANT_API_KEY

    if not url:
        # Fallback to local logic if needed, but usually Qdrant needs URL/Client
        # If we are using local logic in setup_vector_store which implicitly assumed client..
        # Let's see: setup_vector_store uses QdrantClient(url=url...) if url else ValueError
        # So we must have a url. 
        # Wait, older code logic for local path?
        # vector_store.py line 219: if url: client = ... else: raise ValueError
        # So URL is mandatory.
        logger.error("Qdrant URL is missing in configuration.")
        raise ValueError("Qdrant URL is required.")

    logger.info(f"Connecting to Qdrant Server at {url}")
    client = QdrantClient(url=url, api_key=api_key)
    
    if not client.collection_exists(collection_name):
         logger.warning(f"Qdrant collection '{collection_name}' does not exist.")
         return None

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )
    return vector_store

def get_vector_store(
    embedding_config: dict = None, 
    collection_name: str = "default_collection",
    db_type: str = "FAISS",
    db_config: dict = None
):
    """
    Retrieves an existing vector store.
    
    Args:
        embedding_config: Configuration for embedding model
        collection_name: Name of the collection/index
        db_type: Type of vector database ("FAISS", "Qdrant")
        db_config: Additional configuration (e.g., Qdrant URL/API Key)
    
    Returns:
        Vector store object or None if not found.
    """
    logger.info(f"Retrieving vector store: {collection_name} (type: {db_type})")
    
    if db_type not in SUPPORTED_VECTOR_DBS:
        raise ValueError(f"Unsupported vector DB type: {db_type}. Supported: {SUPPORTED_VECTOR_DBS}")

    # Get embedding model
    embedding_model = _get_embedding_model(embedding_config)
    
    if db_type == "FAISS":
        return _load_faiss(embedding_model, collection_name, embedding_config)
    elif db_type == "Qdrant":
        return _load_qdrant(embedding_model, collection_name, db_config, embedding_config)
    
    return None
