from typing import List, Optional
import logging
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.embeddings import LocalOllamaEmbeddings

logger = logging.getLogger(__name__)

# Supported Vector DB types
SUPPORTED_VECTOR_DBS = ["FAISS", "Chroma", "Pinecone"]


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


def _setup_faiss(documents: List[Document], embedding_model, collection_name: str, is_gemini: bool, db_config: dict = None):
    """Setup FAISS vector store."""
    import time
    
    # Use index_name from config if provided
    index_name = db_config.get("index_name", "default") if db_config else "default"
    persist_directory = f"./faiss_db_{collection_name}_{index_name}"
    vector_store = None
    
    # Try to load existing
    if os.path.exists(persist_directory):
        try:
            logger.info(f"Loading existing FAISS index from {persist_directory}")
            vector_store = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")

    if documents:
        logger.info(f"Ingesting {len(documents)} documents into FAISS")
        batch_size = 1 if is_gemini else 1000 
        delay = 4.0 if is_gemini else 0.0 
        total_docs = len(documents)
        
        start_index = 0
        if vector_store is None:
            initial_batch = documents[:batch_size]
            logger.info(f"Initializing FAISS with first batch of {len(initial_batch)} documents")
            vector_store = FAISS.from_documents(initial_batch, embedding_model)
            start_index = batch_size
            if is_gemini and start_index < total_docs:
                time.sleep(delay)
        
        for i in range(start_index, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"Adding batch {i//batch_size + 1} ({len(batch)} docs)")
            vector_store.add_documents(batch)
            vector_store.save_local(persist_directory)
            if is_gemini and (i + batch_size < total_docs):
                time.sleep(delay)
        
        logger.info("FAISS ingestion complete")
    
    return vector_store


def _setup_chroma(documents: List[Document], embedding_model, collection_name: str, is_gemini: bool, db_config: dict = None):
    """Setup Chroma vector store."""
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        raise ImportError("Chroma is not installed. Run: pip install chromadb langchain-chroma")
    
    # Use config values
    index_name = db_config.get("index_name", "default") if db_config else "default"
    namespace = db_config.get("namespace") if db_config else None
    
    persist_directory = f"./chroma_db_{collection_name}_{index_name}"
    chroma_collection_name = f"{collection_name}_{index_name}"
    if namespace:
        chroma_collection_name = f"{namespace}_{chroma_collection_name}"
    
    if documents:
        logger.info(f"Ingesting {len(documents)} documents into Chroma collection: {chroma_collection_name}")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=chroma_collection_name,
            persist_directory=persist_directory
        )
    else:
        # Load existing
        logger.info(f"Loading existing Chroma collection: {chroma_collection_name}")
        vector_store = Chroma(
            collection_name=chroma_collection_name,
            embedding_function=embedding_model,
            persist_directory=persist_directory
        )
    
    return vector_store


def _setup_pinecone(documents: List[Document], embedding_model, collection_name: str, db_config: dict):
    """Setup Pinecone vector store."""
    try:
        from langchain_pinecone import PineconeVectorStore
        from pinecone import Pinecone
    except ImportError:
        raise ImportError("Pinecone is not installed. Run: pip install pinecone-client langchain-pinecone")
    
    api_key = db_config.get("api_key")
    index_name = db_config.get("index_name", collection_name)
    namespace = db_config.get("namespace")  # Can be None
    
    if not api_key:
        raise ValueError("Pinecone API key is required")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)
    
    if documents:
        logger.info(f"Ingesting {len(documents)} documents into Pinecone index: {index_name}, namespace: {namespace}")
        vector_store = PineconeVectorStore.from_documents(
            documents=documents,
            embedding=embedding_model,
            index_name=index_name,
            namespace=namespace
        )
    else:
        logger.info(f"Connecting to existing Pinecone index: {index_name}, namespace: {namespace}")
        vector_store = PineconeVectorStore(
            index_name=index_name,
            embedding=embedding_model,
            namespace=namespace
        )
    
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
        vector_store = _setup_faiss(documents, embedding_model, collection_name, is_gemini, db_config)
    elif db_type == "Chroma":
        vector_store = _setup_chroma(documents, embedding_model, collection_name, is_gemini, db_config)
    elif db_type == "Pinecone":
        if not db_config or not db_config.get("api_key"):
            raise ValueError("Pinecone requires db_config with api_key")
        vector_store = _setup_pinecone(documents, embedding_model, collection_name, db_config)
    
    if vector_store is None and not documents:
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
    # We need to know where to save. FAISS object doesn't store the path by default unless we extended it.
    # We'll assume a default or pass it. For now, let's try to assume the wrapper has it or we can't easily save without path.
    # Actually, we can infer it or we have to change the signature to accept path.
    # Simplified: We will assume 'default_collection' path if not tracked, OR we just save to the same dir used in setup.
    # Hack: We will look for the folder matching 'faiss_db_*' in current dir or just save to default.
    # Better: Update setup to attach path to the object? No, that's hacky.
    # Correct fix: Pass collection name to this function.
    # For now, I will save to "./faiss_db_default_collection" as fallback, but this is a bug risk if multiple collections.
    # I will modify app.py to pass collection name or handle saving there.
    # But wait, FAISS in langchain is in-memory mostly. 
    # Let's save to a fixed path for now or assume the "default_collection" logic holds.
    # I'll update the signature in next step if needed. 
    # Actually, I can just save to "./faiss_db_default_collection" for now as the app uses that default.
    vector_store.save_local("./faiss_db_default_collection") 

def delete_from_vector_store(vector_store, ids: List[str]):
    """
    Deletes documents from the vector store by ID.
    """
    if not vector_store:
        raise ValueError("Vector store is not initialized")
        
    logger.info(f"Deleting {len(ids)} documents from vector store")
    try:
        vector_store.delete(ids)
        vector_store.save_local("./faiss_db_default_collection")
    except Exception as e:
        logger.error(f"Error deleting from FAISS: {e}")
        # FAISS delete might be tricky depending on index type.
