
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Central configuration for the application.
    """
    # Qdrant Configuration
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
    
    # Qdrant Optimization
    QDRANT_MEMMAP_THRESHOLD = int(os.getenv("QDRANT_MEMMAP_THRESHOLD", "200000"))
    QDRANT_INDEXING_THRESHOLD = int(os.getenv("QDRANT_INDEXING_THRESHOLD", "20000"))
    QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "32"))
    QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
    QDRANT_HNSW_FULL_SCAN_THRESHOLD = int(os.getenv("QDRANT_HNSW_FULL_SCAN_THRESHOLD", "10000"))
    QDRANT_SEGMENT_NUMBER = int(os.getenv("QDRANT_SEGMENT_NUMBER", "2"))
    QDRANT_WAL_CAPACITY_MB = int(os.getenv("QDRANT_WAL_CAPACITY_MB", "64"))
    QDRANT_WAL_SEGMENTS_AHEAD = int(os.getenv("QDRANT_WAL_SEGMENTS_AHEAD", "2"))

    # API Keys
    # Removed OpenAI and Google keys as requested

    # Local Services
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    LOCAL_EMBEDDING_BASE_URL = os.getenv("LOCAL_EMBEDDING_BASE_URL", "http://localhost:11434")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
