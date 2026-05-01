import os
import fitz
import numpy as np
import faiss
import pickle
import hashlib
import nltk
import torch
import re
import time
import pandas as pd
import docx2txt
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class Retriever:
    def __init__(self, model_name='allenai/specter2_base', rerank_name='cross-encoder/ms-marco-MiniLM-L-6-v2', device=None, index_file="data.index", data_file="data.pkl"):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.embed_model = SentenceTransformer(model_name, device=self.device)
        self.reranker = CrossEncoder(rerank_name, device=self.device)
        self.index = None
        self.bm25 = None
        self.chunks = []
        self.embeddings_cache = []
        
        self.INDEX_FILE = index_file
        self.DATA_FILE = data_file

    def _clean_text(self, text):
        text = str(text)
        text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
        text = re.sub(r'[^\w\s\.\,\?\!\:\-\%\(\)]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text(self, text, max_tokens=350):
        sentences = sent_tokenize(text)
        chunks, current_chunk, current_length = [], [], 0
        for sent in sentences:
            sent_len = len(sent.split())
            if current_length + sent_len > max_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
            current_chunk.append(sent)
            current_length += sent_len
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def _build_bm25(self):
        if not self.chunks:
            self.bm25 = None
            return
        tokenized_corpus = [c['text'].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _build_faiss(self):
        if not self.embeddings_cache:
            self.index = None
            return
        all_vecs = np.array(self.embeddings_cache).astype('float32')
        dim = all_vecs.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(all_vecs)

    def save(self):
        if self.index:
            faiss.write_index(self.index, self.INDEX_FILE)
            with open(self.DATA_FILE, 'wb') as f:
                pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings_cache}, f)

    def load(self):
        if os.path.exists(self.INDEX_FILE) and os.path.exists(self.DATA_FILE):
            self.index = faiss.read_index(self.INDEX_FILE)
            with open(self.DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.embeddings_cache = data["embeddings"]
            self._build_bm25()
            return True
        return False

    def sync(self, folder_path_or_chunks="./pdf"):
        if isinstance(folder_path_or_chunks, list):
            self.chunks = []
            self.embeddings_cache = []
            for c in folder_path_or_chunks:
                text = c['text']
                vecs = self.embed_model.encode([text], normalize_embeddings=True, show_progress_bar=False)
                self.chunks.append({"text": text, "source": c['source']})
                self.embeddings_cache.append(vecs[0].tolist())
            if self.chunks:
                self._build_faiss()
                self._build_bm25()
                self.save()
            return

        folder_path = folder_path_or_chunks
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path)
            return

        IGNORED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.gif', '.mp4', '.zip', '.rar', '.exe', '.pkl', '.index', '.db', '.sqlite', '.pyc', '.DS_Store')
        current_files = {f for f in os.listdir(folder_path) if not f.lower().endswith(IGNORED_EXTENSIONS)}
        indexed_files = {c['source'] for c in self.chunks}
        to_delete = indexed_files - current_files
        to_add = current_files - indexed_files

        if not to_delete and not to_add and self.index:
            return

        needs_rebuild = False
        if to_delete:
            indices = [i for i, c in enumerate(self.chunks) if c['source'] not in to_delete]
            self.chunks = [self.chunks[i] for i in indices]
            self.embeddings_cache = [self.embeddings_cache[i] for i in indices]
            needs_rebuild = True

        if to_add:
            print(f"Indexing {len(to_add)} new files...")
            for file in tqdm(to_add, desc="Indexing"):
                path = os.path.join(folder_path, file)
                text = ""
                try:
                    file_lower = file.lower()
                    if file_lower.endswith(".pdf"):
                        with fitz.open(path) as doc:
                            for page in doc: 
                                text += page.get_text("text") + " "
                    elif file_lower.endswith((".docx", ".doc")):
                        text = docx2txt.process(path)
                    elif file_lower.endswith((".xlsx", ".xls")):
                        df = pd.read_excel(path)
                        text = df.to_string(index=False)
                    elif file_lower.endswith(".csv"):
                        df = pd.read_csv(path)
                        text = df.to_string(index=False)
                    else:
                        try:
                            with open(path, 'r', encoding='utf-8') as f: 
                                text = f.read()
                        except UnicodeDecodeError:
                            continue

                    if text.strip():
                        file_chunks = self.chunk_text(text)
                        if file_chunks:
                            vecs = self.embed_model.encode(file_chunks, normalize_embeddings=True, show_progress_bar=False)
                            for chunk, vec in zip(file_chunks, vecs):
                                self.chunks.append({"text": chunk, "source": file})
                                self.embeddings_cache.append(vec.tolist())
                            needs_rebuild = True
                except Exception as e:
                    print(f"Error processing {file}: {e}")

        if needs_rebuild and self.embeddings_cache:
            self._build_faiss()
            self._build_bm25()
            self.save()

    def mmr(self, candidates, top_k=5, lambda_param=0.45):
        if not candidates: return []
        if len(candidates) <= top_k: return candidates
        candidate_indices = [c['id'] for c in candidates]
        all_embeddings = torch.tensor([self.embeddings_cache[idx] for idx in candidate_indices]).to(self.device).to(torch.float32)
        selected_indices = [0]
        remaining_indices = list(range(1, len(candidates)))
        while len(selected_indices) < top_k and remaining_indices:
            best_score = -float('inf')
            best_idx_in_remaining = -1
            selected_embs = all_embeddings[selected_indices]
            for i in remaining_indices:
                target_emb = all_embeddings[i]
                relevance = candidates[i]['final_score']
                similarities = torch.matmul(selected_embs, target_emb)
                redundancy = torch.max(similarities).item()
                source_penalty = 0
                selected_sources = [candidates[idx]['source'] for idx in selected_indices]
                if candidates[i]['source'] in selected_sources: source_penalty = 0.25
                score = lambda_param * relevance - (1 - lambda_param) * redundancy - source_penalty
                if score > best_score:
                    best_score = score
                    best_idx_in_remaining = i
            if best_idx_in_remaining != -1:
                selected_indices.append(best_idx_in_remaining)
                remaining_indices.remove(best_idx_in_remaining)
            else: break
        return [candidates[i] for i in selected_indices]

    def compress_context(self, text, chunk_id, query_emb, max_sents=4):
        if len(text.split()) < 60: return text
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sents: return text
        s_embs = self.embed_model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.matmul(s_embs, query_emb).tolist()
        ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        top_sentences = ranked[:max_sents]
        sentence_to_idx = {sent: i for i, sent in enumerate(sentences)}
        top_sentences.sort(key=lambda x: sentence_to_idx[x[0]])
        return " ".join([s[0] for s in top_sentences])

    def summarize(self, results, query, token_limit=500):
        if not results: return ""
        all_sentences, sentence_metadata = [], []
        for r in results:
            sents = sent_tokenize(r['text'])
            all_sentences.extend(sents)
            sentence_metadata.extend([r['source']] * len(sents))
        if not all_sentences: return ""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        s_embs = self.embed_model.encode(all_sentences, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.matmul(s_embs, q_emb.T).reshape(-1).tolist()
        ranked = sorted(zip(all_sentences, scores, sentence_metadata), key=lambda x: x[1], reverse=True)
        summary_parts, current_tokens, seen_sources = [], 0, set()
        for sent, score, source in ranked:
            tokens = len(sent.split())
            if current_tokens + tokens > token_limit: break
            if source not in seen_sources:
                summary_parts.append(f"[{source}]: {sent}")
                seen_sources.add(source)
            else: summary_parts.append(sent)
            current_tokens += tokens
        return self._clean_text(" ".join(summary_parts))

    def retrieve(self, query, top_k=5, threshold=-4.5):
        if self.index is None or self.bm25 is None: return [], ""
        sub_queries = [query]
        if ',' in query or ' and ' in query.lower():
            parts = re.split(r',| and ', query, flags=re.IGNORECASE)
            sub_queries.extend([p.strip() for p in parts if len(p.strip()) > 3])
        candidate_pool = {}
        q_vecs = self.embed_model.encode(sub_queries, normalize_embeddings=True, show_progress_bar=False)
        for i, q_text in enumerate(sub_queries):
            q_vec = q_vecs[i].reshape(1, -1).astype('float32')
            v_scores, v_indices = self.index.search(q_vec, 60)
            t_q = q_text.lower().split()
            b_scores = self.bm25.get_scores(t_q)
            b_indices = np.argsort(b_scores)[::-1][:60]
            for score, idx in zip(v_scores[0], v_indices[0]):
                if idx != -1:
                    cid = int(idx)
                    if cid not in candidate_pool: 
                        cand = self.chunks[cid].copy()
                        cand['id'] = cid
                        candidate_pool[cid] = {**cand, "v_s": float(score), "b_s": 0.0}
                    else: candidate_pool[cid]["v_s"] = max(candidate_pool[cid]["v_s"], float(score))
            for idx in b_indices:
                cid = int(idx)
                if cid in candidate_pool: candidate_pool[cid]["b_s"] = max(candidate_pool[cid]["b_s"], float(b_scores[idx]))
                else:
                    cand = self.chunks[cid].copy()
                    cand['id'] = cid
                    candidate_pool[cid] = {**cand, "v_s": 0.0, "b_s": float(b_scores[idx])}
        candidates = list(candidate_pool.values())
        if not candidates: return [], ""
        candidates.sort(key=lambda x: (0.4 * x['v_s']) + (0.6 * (x['b_s']/20.0)), reverse=True)
        rerank_set = candidates[:100]
        pairs = [[query, c['text']] for c in rerank_set]
        r_scores = self.reranker.predict(pairs, show_progress_bar=False)
        for i, c in enumerate(rerank_set):
            c["final_score"] = (0.2 * c['v_s']) + (0.2 * (c['b_s']/20.0)) + (0.6 * float(r_scores[i]))
        rerank_set.sort(key=lambda x: x["final_score"], reverse=True)
        final_set = self.mmr(rerank_set, top_k=top_k)
        for c in final_set:
            c['text'] = self._clean_text(c['text'])
        summary_str = self.summarize(final_set, query, token_limit=500)
        return final_set, summary_str
