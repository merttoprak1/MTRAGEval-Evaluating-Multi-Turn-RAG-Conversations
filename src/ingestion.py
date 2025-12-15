import json
import logging
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def load_json_documents(file_path: str) -> List[Document]:
    """
    Parses JSON or JSONL files.
    Supports standard JSON list of docs or JSONL (one doc per line).
    """
    logger.info(f"Loading documents from {file_path}")
    documents = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Check if file is JSONL by trying to parse first line
            first_line = f.readline()
            f.seek(0)
            
            is_jsonl = False
            try:
                json.loads(first_line)
                # If the file has multiple lines and the first line is valid JSON, 
                # it's likely JSONL (or a pretty-printed JSON, but we'll assume JSONL if it parses line by line)
                # A robust check would be to see if the whole file parses as a list.
                # Let's try to load as full JSON first, if that fails, try JSONL.
                pass
            except:
                pass

        # Try loading as standard JSON first
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, dict) and "documents" in data and isinstance(data["documents"], list):
                # Format 1: {"documents": [...]}
                logger.debug("Detected JSON format: {'documents': [...]}")
                for doc_data in data["documents"]:
                    page_content = f"Title: {doc_data.get('title', '')}\n\n{doc_data.get('text', '')}"
                    metadata = {k: v for k, v in doc_data.items() if k not in ['text']}
                    documents.append(Document(page_content=page_content, metadata=metadata))
            elif isinstance(data, list):
                # Format 2: List of docs
                logger.debug("Detected JSON format: List of documents")
                for item in data:
                    if isinstance(item, dict):
                         page_content = f"Title: {item.get('title', '')}\n\n{item.get('text', '')}"
                         metadata = {k: v for k, v in item.items() if k not in ['text']}
                         documents.append(Document(page_content=page_content, metadata=metadata))
            
        except json.JSONDecodeError:
            # Failed to load as whole JSON, try JSONL
            is_jsonl = True

        if not documents: # If still empty, try JSONL
             logger.info("Attempting to load as JSONL")
             with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            doc_data = json.loads(line)
                            # Handle new schema keys: _id, id, url, text, title
                            # We map them to metadata, keeping text as content
                            title = doc_data.get('title', '')
                            text = doc_data.get('text', '')
                            page_content = f"Title: {title}\n\n{text}"
                            
                            metadata = {k: v for k, v in doc_data.items() if k not in ['text']}
                            documents.append(Document(page_content=page_content, metadata=metadata))
                        except json.JSONDecodeError:
                            logger.warning(f"Skipping invalid JSON line: {line[:50]}...")
                            continue

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}", exc_info=True)
        return []

def chunk_documents(documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """
    Splits documents into smaller chunks for vector storage.
    """
    logger.info(f"Chunking {len(documents)} documents with size={chunk_size}, overlap={chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Generated {len(chunks)} chunks")
    return chunks
