import os
import re
import json
import pickle
import hashlib
import numpy as np
import faiss
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

from llm_ollama import OllamaLLM
from pdf_parser import parse_pdf_to_text

INDEX_FILE = "data.index"
DATA_FILE = "data.pkl"


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _chunk_text(text: str, max_tokens: int = 400) -> list[str]:
    sentences = sent_tokenize(text)
    chunks, buf, buf_len = [], [], 0
    for s in sentences:
        n = len(s.split())
        if buf_len + n > max_tokens and buf:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(s)
        buf_len += n
    if buf:
        chunks.append(" ".join(buf))
    return chunks


class Retriever:
    def __init__(self, model: str = "llama3"):
        self.llm = OllamaLLM(model)
        self.index: faiss.Index | None = None
        self.chunks: list[dict] = []
        self.embeddings: list[list[float]] = []

    def _embed(self, text: str) -> np.ndarray:
        vec = self.llm.embed(text)
        arr = np.array(vec, dtype="float32")
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr /= norm
        return arr

    def _build_index(self):
        if not self.embeddings:
            self.index = None
            return
        mat = np.array(self.embeddings, dtype="float32")
        dim = mat.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(mat)

    def save(self):
        if self.index is None:
            return
        faiss.write_index(self.index, INDEX_FILE)
        with open(DATA_FILE, "wb") as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)

    def load(self) -> bool:
        if not (os.path.exists(INDEX_FILE) and os.path.exists(DATA_FILE)):
            return False
        self.index = faiss.read_index(INDEX_FILE)
        with open(DATA_FILE, "rb") as f:
            data = pickle.load(f)
        self.chunks = data["chunks"]
        self.embeddings = data["embeddings"]
        return True

    def index_text(self, text: str, source: str):
        raw_chunks = _chunk_text(_clean(text))
        for chunk in raw_chunks:
            vec = self._embed(chunk)
            self.chunks.append({"text": chunk, "source": source})
            self.embeddings.append(vec.tolist())
        self._build_index()
        self.save()

    def index_pdf(self, pdf_path: str, output_dir: str = "./output"):
        text = parse_pdf_to_text(pdf_path, output_dir)
        self.index_text(text, source=Path(pdf_path).name)

    def index_folder(self, folder: str = "./pdf", output_dir: str = "./output"):
        folder = os.path.abspath(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)
            return
        indexed = {c["source"] for c in self.chunks}
        for fname in os.listdir(folder):
            if fname in indexed:
                continue
            fpath = os.path.join(folder, fname)
            try:
                if fname.lower().endswith(".pdf"):
                    self.index_pdf(fpath, output_dir)
                elif fname.lower().endswith((".txt", ".md")):
                    with open(fpath, "r", encoding="utf-8") as f:
                        self.index_text(f.read(), source=fname)
            except Exception as e:
                print(f"[Retriever] Error indexing {fname}: {e}")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if self.index is None or not self.chunks:
            return []
        q_vec = self._embed(query).reshape(1, -1)
        scores, indices = self.index.search(q_vec, min(top_k * 3, len(self.chunks)))
        results = []
        seen = set()
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1 or idx in seen:
                continue
            seen.add(idx)
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def answer(self, query: str, top_k: int = 5) -> dict:
        chunks = self.retrieve(query, top_k)
        if not chunks:
            return {"answer": "No relevant context found.", "sources": [], "chunks": []}
        context = "\n\n".join(
            f"[{c['source']}]: {c['text']}" for c in chunks
        )
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        answer = self.llm.generate(prompt)
        return {
            "answer": answer,
            "sources": list({c["source"] for c in chunks}),
            "chunks": chunks,
        }
