from typing import List
import logging
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# Using OpenAIEmbeddings by default, but this could be swapped for HuggingFaceEmbeddings for full local support
# For now, we'll stick to OpenAIEmbeddings as it's standard, but we can make this configurable if needed.

from src.embeddings import LocalOllamaEmbeddings

logger = logging.getLogger(__name__)

def setup_vector_store(documents: List[Document], embedding_config: dict = None, collection_name: str = "default_collection"):
    """
    Initializes ChromaDB and adds documents.
    Supports persistence and collections.
    """
    logger.info(f"Setting up vector store with collection: {collection_name}")
    embedding_model = None
    
    if embedding_config:
        provider = embedding_config.get("provider", "OpenAI")
        logger.info(f"Using embedding provider: {provider}")
        
        if provider == "OpenAI":
            api_key = embedding_config.get("api_key")
            if not api_key:
                logger.error("API Key missing for OpenAI embeddings")
                raise ValueError("API Key is required for OpenAI embeddings.")
            embedding_model = OpenAIEmbeddings(api_key=api_key)
            
        elif provider == "Local":
            base_url = embedding_config.get("base_url")
            model_name = embedding_config.get("model_name")
            logger.info(f"Configuring Local embeddings: URL={base_url}, Model={model_name}")
            if not base_url or not model_name:
                logger.error("Base URL or Model Name missing for Local embeddings")
                raise ValueError("Base URL and Model Name are required for Local embeddings.")
            
            # Use our custom class
            embedding_model = LocalOllamaEmbeddings(base_url=base_url, model=model_name)
    
    if embedding_model is None:
        # Fallback default
        logger.warning("No embedding config provided, falling back to OpenAI default")
        embedding_model = OpenAIEmbeddings()

    # Persist directory
    persist_directory = "./chroma_db"
    logger.info(f"Using persist directory: {persist_directory}")

    if documents:
        # If documents are provided, we are ingesting/adding to the store
        logger.info(f"Ingesting {len(documents)} documents into collection '{collection_name}'")
        logger.debug(f"Initializing Chroma with documents. Persist directory: {persist_directory}")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        logger.info("Ingestion complete")
    else:
        # If no documents, we are just loading the existing store
        logger.info(f"Loading existing collection '{collection_name}'")
        logger.debug(f"Initializing Chroma without documents. Persist directory: {persist_directory}")
        vector_store = Chroma(
            embedding_function=embedding_model,
            collection_name=collection_name,
            persist_directory=persist_directory
        )
        
    return vector_store

def get_retriever(vector_store, k: int = 4):
    """
    Returns a retriever object from the vector store.
    """
    logger.info(f"Creating retriever with k={k}")
    return vector_store.as_retriever(search_kwargs={"k": k})

def add_to_vector_store(vector_store, documents: List[Document]):
    """
    Adds documents to an existing vector store.
    """
    logger.info(f"Adding {len(documents)} documents to vector store")
    vector_store.add_documents(documents)
    vector_store.persist()

def delete_from_vector_store(vector_store, ids: List[str]):
    """
    Deletes documents from the vector store by ID.
    """
    logger.info(f"Deleting {len(ids)} documents from vector store")
    vector_store.delete(ids=ids)
    vector_store.persist()
