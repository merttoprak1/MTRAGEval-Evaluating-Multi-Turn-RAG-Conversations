import unittest
import os
import sys
# Add the project root to the path so we can import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ingestion import load_json_documents

class TestIngestion(unittest.TestCase):
    def test_load_json_documents(self):
        file_path = "sample_data.json"
        documents = load_json_documents(file_path)
        
        self.assertEqual(len(documents), 2)
        self.assertIn("2017 Arizona Cardinals season", documents[0].page_content)
        self.assertEqual(documents[0].metadata["document_id"], "822086267_6698-7277-0-579")

if __name__ == '__main__':
    unittest.main()
