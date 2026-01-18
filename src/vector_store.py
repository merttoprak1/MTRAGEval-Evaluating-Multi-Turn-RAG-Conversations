import logging
import os
import time
from typing import List, Optional
from time import perf_counter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import Distance, VectorParams
from src.embeddings import LocalOllamaEmbeddings
from src.file_manager import FileManager
import time
from qdrant_client.models import (
    VectorParams,
    Distance,
    HnswConfigDiff,
    OptimizersConfigDiff,
    WalConfigDiff,
)
import json
from src.config import Config
logger = logging.getLogger(__name__)

# Supported Vector DB types
SUPPORTED_VECTOR_DBS = ["FAISS", "Qdrant"]


def _get_embedding_model(embedding_config: dict):
    """
    Factory function to create embedding model based on config.
    """
    if not embedding_config:
        logger.warning("No embedding config provided, falling back to Local Default")
        # Default fallback if nothing provided - assume local
        return LocalOllamaEmbeddings(base_url=Config.LOCAL_EMBEDDING_BASE_URL, model="nomic-embed-text")
    
    provider = embedding_config.get("provider", "Local")
    logger.info(f"Using embedding provider: {provider}")
    
    if provider == "Local":
        base_url = embedding_config.get("base_url")
        model_name = embedding_config.get("model_name")
        logger.info(f"Configuring Local embeddings: URL={base_url}, Model={model_name}")
        if not base_url or not model_name:
            logger.error("Base URL or Model Name missing for Local embeddings")
            raise ValueError("Base URL and Model Name are required for Local embeddings.")
        return LocalOllamaEmbeddings(base_url=base_url, model=model_name)
    
    else:
        # Fallback or Error for removed providers
        if provider in ["OpenAI", "Gemini"]:
             raise ValueError(f"Provider '{provider}' has been removed/disabled.")
        raise ValueError(f"Unsupported embedding provider: {provider}")



def _batched_add_documents(
    vector_store, 
    documents: List[Document], 
    batch_size: int = 100, 
    delay: float = 0.0,
    is_faiss: bool = False,
    persist_directory: str = None
):
    """
    Helper to add documents in batches with optional delay and logging.
    """
    if not documents:
        return

    total_docs = len(documents)
    logger.info(f"Starting ingestion of {total_docs} documents (Batch size: {batch_size})")
    
    start_time = perf_counter()
    
    for i in range(0, total_docs, batch_size):
        batch = documents[i:i + batch_size]
        logger.info(f"Ingesting batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size} ({len(batch)} docs)")
        
        try:
            vector_store.add_documents(batch)
            
            # Immediate save for FAISS to avoid data loss on crash if large ingestion
            if is_faiss and persist_directory:
                vector_store.save_local(persist_directory)

            if delay > 0 and (i + batch_size < total_docs):
                time.sleep(delay)
        except Exception as e:
            logger.error(f"Error ingesting batch {i // batch_size + 1}: {e}", exc_info=True)
            # Continuation choice: continue to next batch or raise?
            # User wants robust logging, usually implies non-stop or at least visible error. 
            # We'll log and continue to try to salvage remaining batches.
            
    elapsed = perf_counter() - start_time
    logger.info(f"Ingestion complete. Processed {total_docs} documents in {elapsed:.2f} seconds.")


def _setup_faiss(
    documents: List[Document],
    embedding_model,
    collection_name: str,
    db_config: dict = None,
    embedding_config: dict = None
):
    """
    FAISS collection behavior:
    - If collection exists → load FAISS → append documents
    - If collection does not exist → create → build FAISS
    """
    model_name = embedding_config.get("model_name", "default_model") if embedding_config else "default_model"
    persist_directory = FileManager.get_collection_path("faiss", model_name, collection_name)
    FileManager.ensure_directory(persist_directory)
    
    index_faiss_path = os.path.join(persist_directory, "index.faiss")
    index_pkl_path = os.path.join(persist_directory, "index.pkl")

    batch_size = 1000
    delay = 0.0

    vector_store = None
    existing_index = os.path.exists(index_faiss_path) and os.path.exists(index_pkl_path)
    
    start_time = perf_counter()

    # 1. Load or Initialize
    if existing_index:
        logger.info(f"Loading existing FAISS collection: {collection_name} from {persist_directory}")
        try:
            vector_store = FAISS.load_local(
                persist_directory,
                embedding_model,
                allow_dangerous_deserialization=True
            )
            logger.info(f"Loaded {vector_store.index.ntotal} existing vectors")
        except Exception as e:
            logger.error(f"Failed to load existing FAISS index: {e}", exc_info=True)
            raise
            
    elif documents:
        logger.info(f"Creating new FAISS collection: {collection_name}")
        # Initialize with first batch to create the store object
        initial_batch = documents[:batch_size]
        remaining_docs = documents[batch_size:]
        
        vector_store = FAISS.from_documents(initial_batch, embedding_model)
        vector_store.save_local(persist_directory)
        
        documents = remaining_docs # Update documents list to process only remainder
        
        # Save metadata info
        bridge_config = {
            "collection": {
                "vector_db_type": "FAISS", 
                "collection_name": collection_name
            },
            "embedding": embedding_config or {}
        }
        with open(f"{persist_directory}/info.json", "w", encoding="utf-8") as f:
            json.dump(bridge_config, f, ensure_ascii=False, indent=4)
            
    else:
        logger.warning(f"Collection '{collection_name}' does not exist and no documents provided.")
        return None

    # 2. Add remaining/new documents
    if documents:
        _batched_add_documents(
            vector_store, 
            documents, 
            batch_size=batch_size, 
            delay=delay, 
            is_faiss=True, 
            persist_directory=persist_directory
        )
    
    # Ensure persist directory is tracked
    if vector_store:
        vector_store._persist_directory = persist_directory

    elapsed = perf_counter() - start_time
    logger.info(f"FAISS setup for '{collection_name}' took {elapsed:.2f}s")
    
    return vector_store

def _setup_qdrant(
    documents: List[Document],
    embedding_model,
    collection_name: str,
    db_config: dict = None,
    embedding_config: dict = None
):
    # 1. Configuration
    db_config = db_config or {}
    embedding_config = embedding_config or {}

    qdrant_config = db_config.get("collection", db_config)

    url = qdrant_config.get("url") or Config.QDRANT_URL
    api_key = qdrant_config.get("api_key") or Config.QDRANT_API_KEY

    model_name = embedding_config.get("model_name", "default_model")
    local_path = FileManager.get_qdrant_storage_path(model_name)
    FileManager.ensure_directory(local_path)

    # 2. Initialize Client
    if url:
        logger.info(f"Connecting to Qdrant Server at {url}")
        try:
            client = QdrantClient(url=url, api_key=api_key)
        except Exception as e:
             logger.error(f"Failed to connect to Qdrant at {url}: {e}", exc_info=True)
             raise
    else:
        raise ValueError

    # 3. Create Collection (100k+ points optimized)
    # Generate composite name for Qdrant server to distinguish same collection name with different embeddings
    # Sanitized for Qdrant (alphanumeric, -, _)
    # Extra sanitization for model names like "all-minilm:l6-v2"
    sanitized_model = FileManager._sanitize(model_name).replace(" ", "_").replace("/", "_").replace(":", "_")
    sanitized_collection = FileManager._sanitize(collection_name).replace(" ", "_").replace(":", "_")
    
    # Check if we should append model name (avoid double appending if already there)
    if sanitized_model not in sanitized_collection:
        qdrant_collection_name = f"{sanitized_collection}_{sanitized_model}"
    else:
        qdrant_collection_name = sanitized_collection
        
    logger.info(f"Using Qdrant Collection Name: {qdrant_collection_name}")

    if not client.collection_exists(qdrant_collection_name):
        logger.info(f"Collection '{qdrant_collection_name}' not found. Creating...")

        try:
            dummy_vec = embedding_model.embed_query("test")
            dim = len(dummy_vec)
            logger.info(f"Detected embedding dimension: {dim}")
        except Exception as e:
            logger.warning(
                f"Could not detect dimension, defaulting to 1536. Error: {e}",
                exc_info=True
            )
            dim = 1536

        client.create_collection(
            collection_name=qdrant_collection_name,
            vectors_config=VectorParams(
                size=dim,
                distance=Distance.COSINE
            ),
            hnsw_config=HnswConfigDiff(
                m=Config.QDRANT_HNSW_M,                     # graph connectivity (higher = better recall)
                ef_construct=Config.QDRANT_HNSW_EF_CONSTRUCT,         # build-time accuracy
                full_scan_threshold=Config.QDRANT_HNSW_FULL_SCAN_THRESHOLD
            ),
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=Config.QDRANT_INDEXING_THRESHOLD,     # index after enough points
                memmap_threshold=Config.QDRANT_MEMMAP_THRESHOLD,      # Keep 100k+ vectors in RAM for speed
                default_segment_number=Config.QDRANT_SEGMENT_NUMBER      # balanced segments
            ),
            wal_config=WalConfigDiff(
                wal_capacity_mb=Config.QDRANT_WAL_CAPACITY_MB,
                wal_segments_ahead=Config.QDRANT_WAL_SEGMENTS_AHEAD
            )
        )

        # 4. UI Bridge
        ui_persist_dir = FileManager.get_collection_path(
            "qdrant", model_name, collection_name
        )
        FileManager.ensure_directory(ui_persist_dir)

        bridge_config = {
            "collection": {
                "vector_db_type": "Qdrant",
                "collection_name": qdrant_collection_name, 
                "original_name": collection_name 
            },
            "embedding": embedding_config
        }

        with open(
            f"{ui_persist_dir}/info.json",
            "w",
            encoding="utf-8"
        ) as f:
            json.dump(bridge_config, f, ensure_ascii=False, indent=4)

    # 5. Initialize VectorStore
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=qdrant_collection_name,
        embedding=embedding_model,
    )

    # 6. Ingest Documents (batch-safe)
    if documents:
        _batched_add_documents(
            vector_store,
            documents,
            batch_size=256,
            delay=0,
            is_faiss=False
        )
        logger.info("Qdrant Ingestion complete.")

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
    
    # Normalize input to match supported types
    db_type_normalized = None
    for supported in SUPPORTED_VECTOR_DBS:
        if supported.lower() == db_type.lower():
            db_type_normalized = supported
            break
            
    if not db_type_normalized:
        raise ValueError(f"Unsupported vector DB type: {db_type}. Supported: {SUPPORTED_VECTOR_DBS}")
    
    # Use normalized type
    db_type = db_type_normalized
    
    # Get embedding model
    embedding_model = _get_embedding_model(embedding_config)
    
    # Setup appropriate vector store
    if db_type == "FAISS":
        vector_store = _setup_faiss(documents, embedding_model, collection_name, db_config, embedding_config)
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
    
    # Determine settings
    is_faiss = isinstance(vector_store, FAISS)
    persist_dir = getattr(vector_store, '_persist_directory', "./faiss_fallback") if is_faiss else None
    
    # Conservative defaults for generic addition
    batch_size = 200
    delay = 0.0
    
    _batched_add_documents(
        vector_store, 
        documents, 
        batch_size=batch_size, 
        delay=delay,
        is_faiss=is_faiss, 
        persist_directory=persist_dir
    )

    if is_faiss:
        logger.info(f"Final save of FAISS index to {persist_dir}")
        vector_store.save_local(persist_dir)

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
        logger.error(f"Error deleting from Vector Store: {e}", exc_info=True)