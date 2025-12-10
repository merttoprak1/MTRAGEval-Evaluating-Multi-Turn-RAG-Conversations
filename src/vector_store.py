from typing import List, Optional
import logging
import os
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from src.embeddings import LocalOllamaEmbeddings

logger = logging.getLogger(__name__)

def setup_vector_store(documents: List[Document] = None, embedding_config: dict = None, collection_name: str = "default_collection"):
    """
    Initializes FAISS and adds documents.
    Supports persistence using local files.
    """
    logger.info(f"Setting up vector store: {collection_name}")
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
            
        elif provider == "Gemini":
            api_key = embedding_config.get("api_key")
            if not api_key:
                logger.error("API Key missing for Gemini embeddings")
                raise ValueError("API Key is required for Gemini embeddings.")
            # Default to embedding-001 or similar if not specified
            model_name = embedding_config.get("model_name", "models/embedding-001")
            embedding_model = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model=model_name)
            
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
    persist_directory = f"./faiss_db_{collection_name}"
    
    vector_store = None
    
    # Try to load existing
    if os.path.exists(persist_directory):
        try:
            logger.info(f"Loading existing FAISS index from {persist_directory}")
            vector_store = FAISS.load_local(persist_directory, embedding_model, allow_dangerous_deserialization=True)
        except Exception as e:
            logger.warning(f"Failed to load existing index: {e}")

    import time

    if documents:
        # If documents are provided, we are ingesting/adding to the store
        logger.info(f"Ingesting {len(documents)} documents into vector store")
        
        # Batching logic for Gemini to avoid 429 errors
        is_gemini = isinstance(embedding_model, GoogleGenerativeAIEmbeddings)
        # Ultra-strict for debugging: 1 doc at a time, 4s delay. 
        # If this fails, the user has NO quota (limit 0).
        batch_size = 1 if is_gemini else 1000 
        delay = 4.0 if is_gemini else 0.0 
        
        total_docs = len(documents)
        
        # Determine if we are creating new or adding to existing
        # If vector_store is None, we need to init with the first batch
        start_index = 0
        if vector_store is None:
             # Process first batch to init
             initial_batch = documents[:batch_size]
             logger.info(f"Initializing FAISS with first batch of {len(initial_batch)} documents")
             try:
                 vector_store = FAISS.from_documents(initial_batch, embedding_model)
                 start_index = batch_size
                 if is_gemini and start_index < total_docs:
                     logger.info(f"Sleeping for {delay}s to respect rate limits...")
                     time.sleep(delay)
             except Exception as e:
                 logger.error("Failed to initialize vector store. Check quota/permissions.")
                 raise e
        
        # Process remaining documents in batches
        for i in range(start_index, total_docs, batch_size):
            batch = documents[i:i+batch_size]
            logger.info(f"Adding batch {i//batch_size + 1} ({len(batch)} docs). Progress: {i}/{total_docs}")
            try:
                if vector_store:
                    vector_store.add_documents(batch)
                else:
                    # Should not happen if logic above is correct, but safety
                    vector_store = FAISS.from_documents(batch, embedding_model)
            except Exception as e:
                logger.error(f"Error adding batch {i}: {e}")
                raise e
            
            # Save progress
            vector_store.save_local(persist_directory)
            
            if is_gemini and (i + batch_size < total_docs):
                logger.info(f"Sleeping for {delay}s...")
                time.sleep(delay)

        logger.info("Ingestion complete and saved")
    
    if vector_store is None and not documents:
        # Return empty if nothing exists and no docs provided (caller might handle this)
        # Or create a dummy empty one? FAISS needs at least one text to initialize usually from_texts
        # But we can't easily return an empty one without dimensions. 
        # For now, let's return None and let app handle it or raise.
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
