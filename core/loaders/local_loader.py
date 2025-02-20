import os
from langchain_community.document_loaders import PyPDFLoader

class LocalPDFLoader:
    """
    Loader for PDF files from a local folder.
    """
    def __init__(self, directory: str):
        """
        Initialize the local loader with the folder containing PDF files.
        
        Args:
            directory (str): Relative path to the folder containing PDF files.
        """
        self.directory = directory

    def load_documents(self):
        """
        Loads PDF documents from the local folder.
        
        Returns:
            List of tuples (filename, list of Document objects)
        """
        documents = []
        folder_path = os.path.join(os.getcwd(), self.directory)
        if not os.path.exists(folder_path):
            raise ValueError(f"Directory {folder_path} does not exist.")
        
        # Get only files ending with .pdf (case-insensitive)
        pdf_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".pdf")]
        
        for filename in pdf_files:
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                documents.append((filename, docs))
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        return documents
