import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional
import uuid

logger = logging.getLogger(__name__)

DB_FILE = "chat_history.db"

def init_db():
    """
    Initializes the SQLite database with necessary tables.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        
        # Create Sessions Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Create Messages Table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                collection TEXT,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")

def create_session(name: str = None) -> str:
    """
    Creates a new chat session and returns its ID.
    """
    session_id = str(uuid.uuid4())
    if not name:
        name = f"Session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO sessions (id, name, created_at) VALUES (?, ?, ?)",
            (session_id, name, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        logger.info(f"Created new session: {session_id} ({name})")
        return session_id
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return None

def get_sessions() -> List[Dict]:
    """
    Retrieves all sessions, ordered by creation time (newest first).
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return []

def save_message(session_id: str, role: str, content: str, collection: str):
    """
    Saves a message to a specific session.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content, collection, timestamp) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, collection, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error saving message: {e}")

def load_session_history(session_id: str) -> List[Dict]:
    """
    Loads chat history for a specific session.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error loading session history: {e}")
        return []

def delete_session(session_id: str):
    """
    Deletes a session and its messages.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted session: {session_id}")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")

def rename_session(session_id: str, new_name: str):
    """
    Renames a session.
    """
    try:
        conn = sqlite3.connect(DB_FILE)
        cursor = conn.cursor()
        cursor.execute("UPDATE sessions SET name = ? WHERE id = ?", (new_name, session_id))
        conn.commit()
        conn.close()
        logger.info(f"Renamed session {session_id} to '{new_name}'")
    except Exception as e:
        logger.error(f"Error renaming session: {e}")
