import os
import json
import requests
import re
import numpy as np
import faiss
import pickle
import torch
from tqdm import tqdm
from fpdf import FPDF
from pathlib import Path
from nltk.tokenize import sent_tokenize
import nltk
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

from llm_ollama import OllamaLLM

class Retriever:
    def __init__(self, model_name='all-MiniLM-L6-v2', rerank_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.embed_model = SentenceTransformer(model_name, device=self.device)
        self.reranker = CrossEncoder(rerank_name, device=self.device)
        self.index = None
        self.bm25 = None
        self.chunks = []
        self.embeddings = []
        self.index_path = "scifact_data.index"
        self.chunks_path = "scifact_data.pkl"

    def _build_bm25(self):
        if not self.chunks: return
        tok = [c['text'].lower().split() for c in self.chunks]
        self.bm25 = BM25Okapi(tok)

    def _build_faiss(self):
        if not self.embeddings: return
        all_vecs = np.array(self.embeddings).astype('float32')
        self.index = faiss.IndexFlatIP(all_vecs.shape[1])
        self.index.add(all_vecs)

    def add_texts(self, texts, sources):
        new_embs = self.embed_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        for text, source, emb in zip(texts, sources, new_embs):
            self.chunks.append({"text": text, "source": source})
            self.embeddings.append(emb.tolist())
        self._build_faiss()
        self._build_bm25()
        with open(self.chunks_path, 'wb') as f:
            pickle.dump({"chunks": self.chunks, "embeddings": self.embeddings}, f)
        faiss.write_index(self.index, self.index_path)

    def load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.chunks_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.chunks_path, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.embeddings = data["embeddings"]
            self._build_bm25()
            return True
        return False

    def mmr(self, candidates, top_k=3, lambda_param=0.5):
        if not candidates or len(candidates) <= top_k: return candidates
        candidate_indices = [c['idx'] for c in candidates]
        all_embs = np.array([self.embeddings[idx] for idx in candidate_indices])
        selected_indices = [0]
        remaining_indices = list(range(1, len(candidates)))
        while len(selected_indices) < top_k and remaining_indices:
            best_score = -float('inf')
            best_idx = -1
            for i in remaining_indices:
                rel = candidates[i]['final_score']
                sim = np.max([np.dot(all_embs[i], all_embs[idx]) for idx in selected_indices])
                score = lambda_param * rel - (1 - lambda_param) * sim
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx != -1:
                selected_indices.append(best_idx)
                remaining_indices.remove(best_idx)
            else: break
        return [candidates[i] for i in selected_indices]

    def summarize(self, results, query, limit=500):
        if not results: return ""
        all_sents = []
        for r in results:
            all_sents.extend([(s, r['source']) for s in sent_tokenize(r['text'])])
        if not all_sents: return ""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True)
        s_embs = self.embed_model.encode([s[0] for s in all_sents], normalize_embeddings=True)
        scores = np.dot(s_embs, q_emb.T).flatten()
        ranked = sorted(zip(all_sents, scores), key=lambda x: x[1], reverse=True)
        summary, curr_toks = [], 0
        for (sent, src), score in ranked:
            toks = len(sent.split())
            if curr_toks + toks > limit: break
            summary.append(f"[{src}]: {sent}")
            curr_toks += toks
        return " ".join(summary)

    def retrieve(self, query, top_k=3):
        if self.index is None or self.bm25 is None: return [], ""
        q_vec = self.embed_model.encode([query], normalize_embeddings=True).astype('float32')
        v_scores, v_indices = self.index.search(q_vec, 60)
        b_scores = self.bm25.get_scores(query.lower().split())
        candidate_pool = {}
        for s, idx in zip(v_scores[0], v_indices[0]):
            if idx != -1:
                candidate_pool[int(idx)] = {"v_s": float(s), "b_s": float(b_scores[int(idx)])}
        candidates = []
        for idx, s in candidate_pool.items():
            cand = self.chunks[idx].copy()
            cand.update({"idx": idx, "v_s": s["v_s"], "b_s": s["b_s"]})
            candidates.append(cand)
        if not candidates: return [], ""
        candidates.sort(key=lambda x: (0.5 * x['v_s']) + (0.5 * (x['b_s']/20.0)), reverse=True)
        rerank_set = candidates[:40]
        pairs = [[query, c['text']] for c in rerank_set]
        r_scores = self.reranker.predict(pairs, show_progress_bar=False)
        for i, c in enumerate(rerank_set):
            c["final_score"] = (0.2 * c['v_s']) + (0.2 * (c['b_s']/20.0)) + (0.6 * float(r_scores[i]))
        rerank_set.sort(key=lambda x: x["final_score"], reverse=True)
        final_set = self.mmr(rerank_set, top_k=top_k)
        summary = self.summarize(final_set, query)
        return final_set, summary

class RagAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tpl = """You are a General Information Specialist. You must follow these strict requirements for the generation of your response. You must return the output as plain text paragraphs only and avoid all forms of Markdown formatting such as hashtags for headers, asterisks for bold or italic text, and backticks for code. You are prohibited from using bullet points, numbered lists, or any special characters and symbols that would break the flow of a standard text document. Your output must be a sequence of clean, standard paragraphs without any code blocks or technical syntax, and the entire response must be written in English. Your logic must be grounded entirely in the provided context, meaning you should never use external knowledge or your own pre-trained data to answer. If the information requested by the user is not present in the chunks or summary, you must explicitly state that the documents do not contain the answer to avoid any form of hallucination. It is mandatory to cite the source filename for every factual statement by placing the citation in parentheses at the end of the relevant sentence, using the format (Source: filename.pdf). If different sources provide conflicting information, you should report all perspectives while citing their specific origins. You must maintain an objective and analytical tone throughout the response, avoiding speculation or phrases like I think or perhaps. Your response must begin with a direct answer to the query in the first paragraph, followed by several paragraphs of detailed supporting evidence synthesized from the retrieved chunks, ensuring every significant point is properly cited.
The following data is provided for your analysis:
User Query: {}
Consolidated Summary: {}
Retrieved Chunks: {}"""
    def draft(self, q, s, c): return self.llm.generate(self.tpl.format(q, s, c))

class Agent1:
    def __init__(self, llm):
        self.llm = llm
        self.tpl = """You are the Grounding Auditor, an uncompromising and rigorous Fact-Checker for an advanced Retrieval-Augmented Generation (RAG) system. Your sole objective is to evaluate a Draft Answer by cross-referencing it exclusively against the provided Retrieved Chunks. You must enforce zero-tolerance for hallucinations. Your task is to perform sentence-by-sentence Natural Language Inference (NLI). For every claim made in the Draft Answer, you must verify if it is strictly entailed by the Retrieved Chunks. You must also verify that all source citations within the Draft Answer accurately match the contents of that specific chunk. You are strictly forbidden from using external knowledge. You must follow these strict requirements for the generation of your response. You must return the output as plain text paragraphs only and avoid all forms of Markdown formatting such as hashtags for headers, asterisks for bold or italic text, and backticks for code. You are prohibited from using bullet points, numbered lists, or any special characters and symbols that would break the flow of a standard text document. Your output must be a sequence of clean, standard paragraphs without any code blocks or technical syntax, and the entire response must be written in English. In your response, you must explicitly state an overall grounding confidence score as a percentage in the first paragraph. In the following paragraphs, you must clearly identify if there are any hallucinations or citation errors. For any unsupported claim or citation error, you must quote the exact sentence from the Draft Answer, explain in detail why it is unsupported or hallucinated, and suggest how to remove or rewrite it. You must conclude your response with a final paragraph summarizing the overall grounding quality.
The following data is provided for your analysis:
User Query: {}
Retrieved Chunks: {}
Draft Answer: {}"""
    def evaluate(self, q, c, draft): return self.llm.generate(self.tpl.format(q, c, draft))

class Agent2:
    def __init__(self, llm):
        self.llm = llm
        self.tpl = """You are the Logic and Consistency Judge for an advanced Retrieval-Augmented Generation (RAG) system. Your primary responsibility is to evaluate a Draft Answer against the original User Query and the Consolidated Summary of the context. Your task involves three dimensions of analysis. First, completeness to ensure the Draft Answer fully addresses all nuances of the User Query. Second, summary alignment to verify the Draft Answer aligns with the core themes in the Consolidated Summary. Third, internal consistency to check for any logical contradictions within the Draft Answer itself. You must follow these strict requirements for the generation of your response. You must return the output as plain text paragraphs only and avoid all forms of Markdown formatting such as hashtags for headers, asterisks for bold or italic text, and backticks for code. You are prohibited from using bullet points, numbered lists, or any special characters and symbols that would break the flow of a standard text document. Your output must be a sequence of clean, standard paragraphs without any code blocks or technical syntax, and the entire response must be written in English. In your response, you must explicitly state a logic confidence score as a percentage in the first paragraph. In the subsequent paragraphs, you must clearly describe whether the draft is complete and consistent. If there is missing critical information from the summary that must be added, you must detail it thoroughly in paragraph form. If there are logical contradictions, you must describe the conflict and provide clear resolution guidance for the final synthesizer.
The following data is provided for your analysis:
User Query: {}
Consolidated Summary: {}
Draft Answer: {}"""
    def evaluate(self, q, s, draft): return self.llm.generate(self.tpl.format(q, s, draft))

class FusionAgent:
    def __init__(self, llm):
        self.llm = llm
        self.tpl = """You are the Executive Synthesizer and Final Editor for an advanced Retrieval-Augmented Generation (RAG) system. Your role is to generate the final, flawless response to the user by revising a Draft Answer based on the structured critique provided by two expert agents, the Grounding Auditor and the Logic Judge. You must follow these strict requirements for the generation of your response. You must return the output as plain text paragraphs only and avoid all forms of Markdown formatting such as hashtags for headers, asterisks for bold or italic text, and backticks for code. You are prohibited from using bullet points, numbered lists, or any special characters and symbols that would break the flow of a standard text document. Your output must be a sequence of clean, standard paragraphs without any code blocks or technical syntax, and the entire response must be written in English. While you should prioritize the provided context, if the information requested by the user is missing or incomplete in the chunks or summary, you are encouraged to use your extensive internal knowledge or external reasoning to provide a comprehensive answer. In such cases, please clearly distinguish between information derived from the documents and information from your general knowledge, while ensuring the overall response remains professional and relevant to the user query. It is mandatory to cite the source filename for every factual statement from the context (Source: filename.pdf). You must maintain an objective and analytical tone throughout. Before generating your final response, you must integrate feedback from the Grounding Auditor and Logic Judge. After synthesizing the final response, you must provide a numerical confidence score from 0 to 100 on its own line at the very end of your response, representing your degree of certainty in the final answer based on the provided evidence and critiques. The final confidence score must be prefixed with 'Confidence Score: ' followed by the number. You MUST also explicitly include the final label 'SUPPORT', 'REFUTES', or 'NOT_ENOUGH_INFO' in your synthesis.
The following data is provided for your synthesis:
User Query: {}
Draft Answer: {}
Grounding Auditor Critique: {}
Logic Judge Critique: {}
Consolidated Summary: {}
Retrieved Chunks: {}"""
    def fuse(self, q, d, e1, e2, s, c): return self.llm.generate(self.tpl.format(q, d, e1, e2, s, c))

def create_pdf(doc_id, title, abstract, out_dir):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    text = f"Title: {title}\n\nAbstract: {' '.join(abstract)}"
    text = text.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 10, text)
    pdf_path = os.path.join(out_dir, f"{doc_id}.pdf")
    pdf.output(pdf_path)
    return pdf_path

def parse_with_mineru(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            resp = requests.post("http://localhost:8000/file_parse", files={"files": f}, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("markdown")
        if not content:
            results = data.get("results", {})
            if isinstance(results, dict):
                for val in results.values():
                    if isinstance(val, dict) and val.get("md_content"):
                        content = val["md_content"]
                        break
        return content
    except: return None

def process_corpus(corpus_file, pdf_dir, md_dir):
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(md_dir, exist_ok=True)
    with open(corpus_file, "r") as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Processing Corpus"):
        doc = json.loads(line)
        doc_id = str(doc["doc_id"])
        md_path = os.path.join(md_dir, f"{doc_id}.md")
        if os.path.exists(md_path): continue
        pdf_path = os.path.join(pdf_dir, f"{doc_id}.pdf")
        if not os.path.exists(pdf_path):
            create_pdf(doc_id, doc["title"], doc["abstract"], pdf_dir)
        md_content = parse_with_mineru(pdf_path)
        if md_content:
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

def build_index(md_dir, retriever):
    if retriever.load(): return retriever
    texts, sources = [], []
    for fname in tqdm(os.listdir(md_dir), desc="Indexing"):
        if fname.endswith(".md"):
            with open(os.path.join(md_dir, fname), "r", encoding="utf-8") as f:
                texts.append(f.read())
                sources.append(fname)
    if texts: retriever.add_texts(texts, sources)
    return retriever

def main():
    corpus_file, claims_file = "scifact_data/corpus.jsonl", "scifact_data/claims_dev.jsonl"
    pdf_dir, md_dir = "scifact_pdfs", "scifact_mds"
    process_corpus(corpus_file, pdf_dir, md_dir)
    retriever = Retriever()
    build_index(md_dir, retriever)
    llm = OllamaLLM()
    rag_agent, agent1, agent2, fusion_agent = RagAgent(llm), Agent1(llm), Agent2(llm), FusionAgent(llm)
    results = []
    with open(claims_file, "r") as f:
        claims = [json.loads(l) for l in f.readlines()]
    for claim_obj in tqdm(claims, desc="Evaluating"):
        q = claim_obj["claim"]
        chunks, summary = retriever.retrieve(q)
        c_text = " ".join([f"{r['source']} {r['text']}" for r in chunks])
        draft = rag_agent.draft(q, summary, c_text)
        e1 = agent1.evaluate(q, c_text, draft)
        e2 = agent2.evaluate(q, summary, draft)
        final = fusion_agent.fuse(q, draft, e1, e2, summary, c_text)
        label = "NOT_ENOUGH_INFO"
        if "SUPPORT" in final.upper(): label = "SUPPORT"
        elif "REFUTES" in final.upper(): label = "REFUTES"
        results.append({"id": claim_obj["id"], "predicted": label, "gold": claim_obj.get("evidence_label", "UNKNOWN")})
        with open("scifact_results.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
