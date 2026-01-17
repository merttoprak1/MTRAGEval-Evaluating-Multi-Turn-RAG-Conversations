
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

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)

    # Local Services
    LOCAL_LLM_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:11434/v1")
    LOCAL_EMBEDDING_BASE_URL = os.getenv("LOCAL_EMBEDDING_BASE_URL", "http://localhost:11434")

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
