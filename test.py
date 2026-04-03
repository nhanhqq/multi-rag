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
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from nltk.tokenize import sent_tokenize

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', rerank_name='cross-encoder/ms-marco-MiniLM-L-6-v2', device=None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Initializing Retriever on {self.device}...")
        self.embed_model = SentenceTransformer(model_name, device=self.device)
        self.reranker = CrossEncoder(rerank_name, device=self.device)
        self.index = None
        self.bm25 = None
        self.chunks = []
        self.embeddings_cache = []
        
        self.INDEX_FILE = "data.index"
        self.DATA_FILE = "data.pkl"

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

    def _get_metadata_score(self, text, query):
        score = 0
        text_lower = text.lower()
        struct_indicators = ["summary", "conclusion", "introduction", "overview", "key", "important", "note"]
        data_indicators = [":", "-", "•", "1.", "2.", "%", "$", "http"]
        for ind in struct_indicators:
            if ind in text_lower: score += 0.1
        for ind in data_indicators:
            if ind in text_lower: score += 0.05
        if re.search(r'\d+', text): score += 0.1
        if len(text.split('\n')) > 2: score += 0.1
        return score

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
            print(f"Index values saved to {self.INDEX_FILE}")

    def load(self):
        if os.path.exists(self.INDEX_FILE) and os.path.exists(self.DATA_FILE):
            self.index = faiss.read_index(self.INDEX_FILE)
            with open(self.DATA_FILE, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.embeddings_cache = data["embeddings"]
            self._build_bm25()
            print("Loaded index from disk.")
            return True
        return False

    def sync(self, folder_path="./pdf"):
        if not os.path.exists(folder_path): 
            os.makedirs(folder_path)
            print(f"Created {folder_path} folder.")
            return

        current_files = {f for f in os.listdir(folder_path) if f.lower().endswith(('.pdf', '.txt'))}
        indexed_files = {c['source'] for c in self.chunks}
        
        to_delete = indexed_files - current_files
        to_add = current_files - indexed_files

        if not to_delete and not to_add and self.index:
            return

        needs_rebuild = False
        if to_delete:
            print(f"Removing old files: {to_delete}")
            indices = [i for i, c in enumerate(self.chunks) if c['source'] not in to_delete]
            self.chunks = [self.chunks[i] for i in indices]
            self.embeddings_cache = [self.embeddings_cache[i] for i in indices]
            needs_rebuild = True

        if to_add:
            print(f"Indexing new files: {to_add}")
            for file in to_add:
                path = os.path.join(folder_path, file)
                text = ""
                try:
                    if file.endswith(".pdf"):
                        with fitz.open(path) as doc:
                            for page in doc: text += page.get_text("text") + " "
                    else:
                        with open(path, 'r', encoding='utf-8') as f: text = f.read()
                    
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

    def mmr(self, candidates, top_k=5, lambda_param=0.6):
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
                
                score = lambda_param * relevance - (1 - lambda_param) * redundancy
                if score > best_score:
                    best_score = score
                    best_idx_in_remaining = i
            
            if best_idx_in_remaining != -1:
                selected_indices.append(best_idx_in_remaining)
                remaining_indices.remove(best_idx_in_remaining)
            else:
                break
                
        return [candidates[i] for i in selected_indices]

    def compress_context(self, text, chunk_id, query_emb, max_sents=3):
        if len(text.split()) < 50: return text
        
        sentences = sent_tokenize(text)
        if len(sentences) <= max_sents: return text
        
        s_embs = self.embed_model.encode(sentences, normalize_embeddings=True, convert_to_tensor=True, show_progress_bar=False)
        scores = torch.matmul(s_embs, query_emb).tolist()
        
        ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)
        top_sentences = ranked[:max_sents]
        
        sentence_to_idx = {sent: i for i, sent in enumerate(sentences)}
        top_sentences.sort(key=lambda x: sentence_to_idx[x[0]])
        
        return " ".join([s[0] for s in top_sentences])

    def retrieve(self, query, top_k=5, fast_mode=True, threshold=-2.5):
        if self.index is None or self.bm25 is None:
            return []

        expanded_queries = [query]
        if not fast_mode:
            expanded_queries.extend([f"technical details of {query}", f"data regarding {query}"])
        
        q_vecs = self.embed_model.encode(expanded_queries, normalize_embeddings=True, show_progress_bar=False)
        query_emb_tensor = torch.from_numpy(q_vecs[0]).to(self.device).to(torch.float32)
        
        candidate_pool = {}
        for i, q_text in enumerate(expanded_queries):
            q_vec = q_vecs[i].reshape(1, -1).astype('float32')
            v_scores, v_indices = self.index.search(q_vec, 30)
            
            t_q = q_text.lower().split()
            b_scores = self.bm25.get_scores(t_q)
            b_indices = np.argsort(b_scores)[::-1][:25]
            
            for score, idx in zip(v_scores[0], v_indices[0]):
                if idx != -1:
                    cid = int(idx)
                    if cid not in candidate_pool: 
                        chunk_data = self.chunks[cid].copy()
                        chunk_data['id'] = cid
                        candidate_pool[cid] = {**chunk_data, "v_s": float(score), "b_s": 0.0}
                    else:
                        candidate_pool[cid]["v_s"] = max(candidate_pool[cid]["v_s"], float(score))
            
            for idx in b_indices:
                cid = int(idx)
                if cid in candidate_pool: 
                    candidate_pool[cid]["b_s"] = max(candidate_pool[cid]["b_s"], float(b_scores[idx]))
                else:
                    chunk_data = self.chunks[cid].copy()
                    chunk_data['id'] = cid
                    candidate_pool[cid] = {**chunk_data, "v_s": 0.0, "b_s": float(b_scores[idx])}
        
        candidates = list(candidate_pool.values())
        if not candidates: return []
        
        candidates.sort(key=lambda x: (0.3 * x['v_s']) + (0.7 * (x['b_s']/20.0)), reverse=True)
        rerank_set = candidates[:35]
        
        pairs = [[query, c['text']] for c in rerank_set]
        r_scores = self.reranker.predict(pairs, show_progress_bar=False)
        
        for i, c in enumerate(rerank_set):
            m_s = self._get_metadata_score(c['text'], query)
            norm_b = c['b_s'] / 20.0
            c["final_score"] = (0.2 * c['v_s']) + (0.1 * norm_b) + (0.6 * float(r_scores[i])) + (0.1 * m_s)
        
        rerank_set.sort(key=lambda x: x["final_score"], reverse=True)
        
        if not rerank_set or rerank_set[0]["final_score"] < threshold:
            return []
            
        final_set = self.mmr(rerank_set, top_k=top_k)
        
        for c in final_set:
            c['text'] = self.compress_context(c['text'], c['id'], query_emb_tensor)
            
        return final_set

def main():
    RAG = Retriever()
    if not RAG.load(): 
        RAG.sync("./pdf")

    print("\n--- RAG Ready ---")
    while True:
        try:
            user_q = input("\nSearch: ").strip()
            if not user_q: continue
            if user_q.lower() in ['exit', 'quit']: break
            
            start = time.time()
            res = RAG.retrieve(user_q, threshold=-3.5) 
            end = time.time()
            
            if not res:
                print("No relevant information found in the documents.")
                continue
                
            print(f"\n--- Results ({end-start:.2f}s) ---")
            for i, r in enumerate(res):
                print(f"[{i+1}] Source: {r['source']} | Score: {r['final_score']:.3f}")
                print(f"Content: {r['text']}\n")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()