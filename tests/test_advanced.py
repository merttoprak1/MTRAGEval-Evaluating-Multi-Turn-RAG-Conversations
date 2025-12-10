import unittest
import os
import sys
import shutil
# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_json_documents
from src.vector_store import setup_vector_store

class TestAdvancedFeatures(unittest.TestCase):
    def setUp(self):
        # Clean up chroma_db before test
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")

    def test_load_jsonl_documents(self):
        file_path = "sample_data.jsonl"
        documents = load_json_documents(file_path)
        
        self.assertEqual(len(documents), 2)
        self.assertIn("French Revolution", documents[0].page_content)
        self.assertEqual(documents[0].metadata["id"], "837799097_6931-7548-0-617")

    def test_vector_store_persistence_and_collections(self):
        file_path = "sample_data.jsonl"
        documents = load_json_documents(file_path)
        
        # Use local config to avoid OpenAI API key error
        embedding_config = {
            "provider": "Local",
            "base_url": "http://localhost:11434",
            "model_name": "shaw/dmeta-embedding-zh-small-q4"
        }
        
        # Create collection 'test_col'
        vs = setup_vector_store(documents, embedding_config=embedding_config, collection_name="test_col")
        # Persist is automatic in new Chroma versions usually, but let's check if we can reload
        
        # Reload without documents
        vs_loaded = setup_vector_store(documents=None, embedding_config=embedding_config, collection_name="test_col")
        
        # Check if we can retrieve
        retriever = vs_loaded.as_retriever()
        docs = retriever.invoke("French Revolution")
        self.assertTrue(len(docs) > 0)

if __name__ == '__main__':
    unittest.main()
