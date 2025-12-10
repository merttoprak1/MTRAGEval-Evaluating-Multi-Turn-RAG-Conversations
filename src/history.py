import json
import os
import logging
from datetime import datetime
from typing import List, Dict

logger = logging.getLogger(__name__)

HISTORY_FILE = "chat_history.json"

def load_chat_history() -> List[Dict]:
    """
    Loads chat history from a JSON file.
    """
    if not os.path.exists(HISTORY_FILE):
        return []
    
    try:
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            history = json.load(f)
            # Ensure it's a list
            if isinstance(history, list):
                return history
            else:
                logger.warning("Chat history file format invalid (not a list), returning empty.")
                return []
    except Exception as e:
        logger.error(f"Error loading chat history: {e}")
        return []

def save_message(role: str, content: str, collection: str):
    """
    Saves a single message to the chat history.
    """
    entry = {
        "timestamp": datetime.now().isoformat(),
        "role": role,
        "content": content,
        "collection": collection
    }
    
    history = load_chat_history()
    history.append(entry)
    
    try:
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error saving chat history: {e}")
