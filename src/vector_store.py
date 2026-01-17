from typing import List, Optional
import logging
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.embeddings import LocalOllamaEmbeddings
from src.file_manager import FileManager
import time
import json 
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


def _setup_faiss(
    documents: List[Document],
    embedding_model,
    collection_name: str,
    is_gemini: bool,
    db_config: dict = None,
    embedding_config: dict = None
):
    """
    FAISS collection behavior:
    - If collection exists → load FAISS → append documents
    - If collection does not exist → create → build FAISS
    """

    model_name = embedding_config.get("model_name", "default_model")
    persist_directory = FileManager.get_collection_path("faiss", model_name, collection_name)
    FileManager.ensure_directory(persist_directory)
    index_faiss_path = os.path.join(persist_directory, "index.faiss")
    index_pkl_path = os.path.join(persist_directory, "index.pkl")

    batch_size = 100 if is_gemini else 1000
    delay = 1.0 if is_gemini else 0.0

    vector_store = None

    # --------------------------------------------------
    # 1. Create collection folder if missing
    # --------------------------------------------------
    os.makedirs(persist_directory, exist_ok=True)

    # --------------------------------------------------
    # 2. Load existing FAISS index if present
    # --------------------------------------------------
    if os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path):
        logger.info(f"Loading existing FAISS collection: {collection_name}")

        vector_store = FAISS.load_local(
            persist_directory,
            embedding_model,
            allow_dangerous_deserialization=True
        )

        logger.info(f"Loaded {vector_store.index.ntotal} existing vectors")

    # --------------------------------------------------
    # 3. Create FAISS if it does not exist
    # --------------------------------------------------
    elif documents:
        logger.info(f"Creating new FAISS collection: {collection_name}")

        vector_store = FAISS.from_documents(
            documents[:batch_size],
            embedding_model
        )

        documents = documents[batch_size:]
        vector_store.save_local(persist_directory)
        
        bridge_config = {
            "collection": {
                "vector_db_type": "FAISS", 
                "collection_name": collection_name
            },
            "embedding": embedding_config or {}
        }
        
        with open(f"{persist_directory}/info.json", "w", encoding="utf-8") as f:
            json.dump(bridge_config, f, ensure_ascii=False, indent=4)
        # ------------------------------------------------------------

    else:
        raise ValueError(
            f"Collection '{collection_name}' does not exist and no documents were provided"
        )

    # --------------------------------------------------
    # 4. Add new documents (append)
    # --------------------------------------------------
    if documents:
        logger.info(f"Adding {len(documents)} new documents to collection")

        for i in range(0, len(documents), batch_size):
            logger.info(f"Processing :  {i}/{len(documents)}")
            batch = documents[i:i + batch_size]
            vector_store.add_documents(batch)

            if is_gemini and i + batch_size < len(documents):
                time.sleep(delay)

        vector_store.save_local(persist_directory)
        logger.info("FAISS collection updated")

        if vector_store:
            vector_store._persist_directory = persist_directory

    return vector_store

def _setup_qdrant(
    documents: List[Document],
    embedding_model,
    collection_name: str,
    db_config: dict = None,
    embedding_config: dict = None
):
    # 1. Configuration
    if db_config and "collection" in db_config:
         qdrant_config = db_config.get("collection", {})
    else:
         qdrant_config = db_config or {}

    url = qdrant_config.get("url")
    api_key = qdrant_config.get("api_key")
    
    model_name = embedding_config.get("model_name", "default_model")
    local_path = FileManager.get_qdrant_storage_path(model_name)
    FileManager.ensure_directory(local_path)

    # 2. Initialize Client
    if url:
        logger.info(f"Connecting to Qdrant Server at {url}")
        client = QdrantClient(url=url, api_key=api_key)
    else:
        logger.info(f"Using Local Qdrant at {local_path}")
        client = QdrantClient(path=local_path)

    # 3. Create Collection
    if not client.collection_exists(collection_name):
        logger.info(f"Collection '{collection_name}' not found. Creating...")
        
        # Auto-detect dimension
        try:
            # We embed a tiny string to get the exact dimension (768, 1536, etc.)
            dummy_vec = embedding_model.embed_query("test")
            dim = len(dummy_vec)
            logger.info(f"Detected embedding dimension: {dim}")
        except Exception as e:
            logger.warning(f"Could not detect dimension, defaulting to 1536. Error: {e}")
            dim = 1536

        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
        )
        
        # 4. Create UI Bridge (The "Fake" FAISS-like folder)
        ui_persist_dir = FileManager.get_collection_path("qdrant", model_name, collection_name)
        FileManager.ensure_directory(ui_persist_dir)
        
        # Save info.json so the UI knows this is a Qdrant DB
        bridge_config = {
            "collection": {"vector_db_type": "Qdrant", "collection_name": collection_name},
            "embedding": embedding_config or {}
        }
        with open(f"{ui_persist_dir}/info.json", "w", encoding="utf-8") as f:
            json.dump(bridge_config, f, ensure_ascii=False, indent=4)

    # 5. Initialize VectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding_model,
    )

    # 6. Ingest Documents
    if documents:
        logger.info(f"Ingesting {len(documents)} documents into Qdrant...")
        vector_store.add_documents(documents)
        logger.info("Ingestion complete.")

    return vector_store


def setup_vector_store(
    documents: List[Document] = None, 
    embedding_config: dict = None, 
    collection_name: str = "default_collection",
    db_type: str = "FAISS",
    db_config: dict = None
):
    """
    Initializes vector store and adds documents.
    
    Args:
        documents: List of documents to ingest
        embedding_config: Configuration for embedding model
        collection_name: Name of the collection/index
        db_type: Type of vector database ("FAISS", "Chroma", "Pinecone")
        db_config: Additional configuration for specific DB (e.g., Pinecone API key)
    
    Returns:
        Initialized vector store object
    """
    logger.info(f"Setting up vector store: {collection_name} (type: {db_type})")
    
    if db_type not in SUPPORTED_VECTOR_DBS:
        raise ValueError(f"Unsupported vector DB type: {db_type}. Supported: {SUPPORTED_VECTOR_DBS}")
    
    # Get embedding model
    embedding_model = _get_embedding_model(embedding_config)
    is_gemini = isinstance(embedding_model, GoogleGenerativeAIEmbeddings)
    
    # Setup appropriate vector store
    if db_type == "FAISS":
        vector_store = _setup_faiss(documents, embedding_model, collection_name, is_gemini, db_config, embedding_config)
    elif db_type == "Qdrant":
        return _setup_qdrant(documents, embedding_model, collection_name, db_config, embedding_config)
    
    if vector_store is None and not documents:
        # Note: Qdrant client persists, so even without docs, if the collection exists, it returns the store.
        # This check is mostly for in-memory FAISS.
        logger.warning("No existing index and no documents provided.")
        return None
        
    return vector_store

def get_retriever(vector_store, k: int = 4):
    """
    Returns a retriever object from the vector store.
    """
    if not vector_store:
         raise ValueError("Vector store is not initialized")
    logger.info(f"Creating retriever with k={k}")
    return vector_store.as_retriever(search_kwargs={"k": k})

def add_to_vector_store(vector_store, documents: List[Document]):
    """
    Adds documents to an existing vector store.
    """
    if not vector_store:
        raise ValueError("Vector store is not initialized")
    
    logger.info(f"Adding {len(documents)} documents to vector store")
    vector_store.add_documents(documents)

    if isinstance(vector_store, FAISS):
        # We check if we attached the path in setup_vector_store
        if hasattr(vector_store, '_persist_directory'):
            logger.info(f"Saving FAISS index to {vector_store._persist_directory}")
            vector_store.save_local(vector_store._persist_directory)
        else:
            logger.warning("FAISS index has no tracked path. Saving to fallback './faiss_fallback'")
            vector_store.save_local("./faiss_fallback")

def delete_from_vector_store(vector_store, ids: List[str]):
    """
    Deletes documents from the vector store by ID.
    """
    if not vector_store:
        raise ValueError("Vector store is not initialized")
        
    logger.info(f"Deleting {len(ids)} documents from vector store")
    try:
        # Qdrant wrapper supports delete
        vector_store.delete(ids)
        
        # FAISS specific saving
        if isinstance(vector_store, FAISS):
            if hasattr(vector_store, '_persist_directory'):
                logger.info(f"Saving FAISS index to {vector_store._persist_directory}")
                vector_store.save_local(vector_store._persist_directory)
            else:
                logger.warning("FAISS index has no tracked path. Cannot save deletion state.")
            
    except Exception as e:
        logger.error(f"Error deleting from Vector Store: {e}")