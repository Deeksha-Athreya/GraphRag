import unittest
import os
from utils.pdf_extractor import extract_from_pdf
from utils.table_extractor import extract_tables
from utils.graph_builder import build_graph
from utils.rag_model import RAGPipeline

class TestDocumentProcessing(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Set up paths for testing
        cls.pdf_path = 'data/ebooks/book1.pdf'
        cls.extracted_text_path = 'data/extracted/book1/text.txt'
        cls.extracted_images_path = 'data/extracted/book1/images'
        cls.table_extracted_path = 'data/extracted/book1/tables'
        cls.graph_path = 'indexes/graph.db'
        
        # Clean up previous test artifacts
        if os.path.exists(cls.extracted_text_path):
            os.remove(cls.extracted_text_path)
        if os.path.exists(cls.extracted_images_path):
            for img_file in os.listdir(cls.extracted_images_path):
                os.remove(os.path.join(cls.extracted_images_path, img_file))
        if os.path.exists(cls.table_extracted_path):
            for table_file in os.listdir(cls.table_extracted_path):
                os.remove(os.path.join(cls.table_extracted_path, table_file))
        if os.path.exists(cls.graph_path):
            os.remove(cls.graph_path)

    def test_pdf_extraction(self):
        """ Test the extraction of text and images from PDF """
        extract_from_pdf(self.pdf_path, self.extracted_text_path, self.extracted_images_path)
        
        # Check if text file is created
        self.assertTrue(os.path.isfile(self.extracted_text_path))
        
        # Check if images are extracted
        image_files = os.listdir(self.extracted_images_path)
        self.assertGreater(len(image_files), 0, "No images were extracted")

    def test_table_extraction(self):
        """ Test the extraction of tables from PDFs """
        extract_tables(self.pdf_path, self.table_extracted_path)
        
        # Check if table files are created
        table_files = os.listdir(self.table_extracted_path)
        self.assertGreater(len(table_files), 0, "No tables were extracted")

    def test_graph_builder(self):
        """ Test building the graph database from extracted content """
        build_graph('data/extracted', self.graph_path)
        
        # Check if graph database is created
        self.assertTrue(os.path.isfile(self.graph_path))

    def test_rag_pipeline(self):
        """ Test querying the RAG model """
        rag_pipeline = RAGPipeline(document_path='data/extracted')
        rag_pipeline.load_and_index_documents()
        rag_pipeline.save_index(self.graph_path)
        
        query_result = rag_pipeline.query("How does the engine work?")
        
        # Check if query result is returned
        self.assertIsNotNone(query_result.answer)
        self.assertGreater(len(query_result.source_nodes), 0, "No source nodes found")

if __name__ == '__main__':
    unittest.main()
