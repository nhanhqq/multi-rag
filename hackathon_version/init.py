from RAG import Retriever

class DataInitializer:
    def __init__(self, pdf_folder="./pdf"):
        self.pdf_folder = pdf_folder
        self.rag = Retriever()

    def init_data(self):
        if not self.rag.load():
            self.rag.sync(self.pdf_folder)
        return self.rag
