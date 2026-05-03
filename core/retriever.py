import os
import faiss
import numpy as np
import pickle
from .llm_ollama import OllamaLLM

class Retriever:
    def __init__(self, index_path="data.index", chunks_path="data.pkl"):
        self.llm = OllamaLLM()
        self.index_path = index_path
        self.chunks_path = chunks_path
        self.index = None
        self.chunks = []
        if os.path.exists(index_path) and os.path.exists(chunks_path):
            self.load()

    def add_texts(self, texts: list[str], sources: list[str]):
        embeddings = []
        for i, text in enumerate(texts):
            emb = self.llm.embed(text)
            embeddings.append(emb)
            self.chunks.append({"text": text, "source": sources[i]})
        
        embeddings_np = np.array(embeddings).astype('float32')
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings_np.shape[1])
        self.index.add(embeddings_np)
        self.save()

    def search(self, query: str, top_k: int = 5):
        if self.index is None:
            return []
        query_emb = np.array([self.llm.embed(query)]).astype('float32')
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                chunk = self.chunks[idx].copy()
                chunk["score"] = float(dist)
                results.append(chunk)
        return results

    def save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.chunks_path, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
