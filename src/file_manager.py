import os
import shutil
from typing import List

class FileManager:
    BASE_COLLECTIONS_DIR = "collections"
    BASE_QDRANT_STORAGE = "qdrant_local_storage"

    @staticmethod
    def _sanitize(name: str) -> str:
        """Sanitize names to be filesystem safe."""
        return name.replace("/", "_").replace("\\", "_").strip()

    @staticmethod
    def get_collection_path(db_type: str, embedding_model_name: str, collection_name: str) -> str:
        """
        Returns the path for the UI bridge/FAISS index.
        Format: collections/{db_type}/{embedding_model}/{collection_name}
        """
        db = FileManager._sanitize(db_type.lower())
        model = FileManager._sanitize(embedding_model_name)
        col = FileManager._sanitize(collection_name)
        
        path = os.path.join(FileManager.BASE_COLLECTIONS_DIR, db, model, col)
        return path

    @staticmethod
    def get_qdrant_storage_path(embedding_model_name: str) -> str:
        """
        Returns the isolated storage path for Qdrant based on the embedding model.
        Format: qdrant_local_storage/{embedding_model}
        """
        model = FileManager._sanitize(embedding_model_name)
        path = os.path.join(FileManager.BASE_QDRANT_STORAGE, model)
        return path

    @staticmethod
    def ensure_directory(path: str):
        """Creates the directory if it does not exist."""
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def list_collections(db_type: str, embedding_model_name: str) -> List[str]:
        """Lists available collections for a specific DB and Model configuration."""
        path = os.path.join(
            FileManager.BASE_COLLECTIONS_DIR, 
            FileManager._sanitize(db_type.lower()), 
            FileManager._sanitize(embedding_model_name)
        )
        if not os.path.exists(path):
            return []
        
        # Return only directories
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]