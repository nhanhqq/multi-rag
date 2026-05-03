import os
import json
import pandas as pd
import pickle
import faiss
import numpy as np
import fitz
import re
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
from llm_ollama import LLM
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class SimpleRetriever:
    def __init__(self, index_file="traditional_rag.index", data_file="traditional_rag.pkl", folder_path="./data/sailor_contexts", model_name="allenai/specter2_base"):
        self.index_file = index_file
        self.data_file = data_file
        self.model = SentenceTransformer(model_name)
        self.chunks = []
        self.embeddings = []

        if os.path.exists(index_file) and os.path.exists(data_file):
            print(f"Loading existing traditional index: {index_file}")
            self.index = faiss.read_index(index_file)
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
        else:
            print(f"Creating NEW traditional index from {folder_path}...")
            self._build_index(folder_path)

    def _build_index(self, folder_path):
        files = [f for f in os.listdir(folder_path) if f.endswith(('.txt', '.pdf', '.md', '.csv'))]
        all_chunks = []
        for file in tqdm(files, desc="Reading files"):
            path = os.path.join(folder_path, file)
            text = ""
            if file.endswith(".pdf"):
                try:
                    with fitz.open(path) as doc:
                        for page in doc: text += page.get_text() + " "
                except: continue
            else:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            
            if text.strip():
                # Simple chunking: 300 words per chunk
                words = text.split()
                for i in range(0, len(words), 300):
                    chunk_text = " ".join(words[i:i+300])
                    all_chunks.append({"text": chunk_text, "source": file})
        
        print(f"Encoding {len(all_chunks)} chunks...")
        self.chunks = all_chunks
        texts = [c['text'] for c in self.chunks]
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(self.index, self.index_file)
        with open(self.data_file, 'wb') as f:
            pickle.dump({"chunks": self.chunks}, f)
        print("Traditional index created and saved.")

    def retrieve(self, query, top_k=5):
        q_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False).astype('float32')
        scores, indices = self.index.search(q_vec, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.chunks[idx])
        return results

def process_item_traditional_rag(item, rag, llm):
    question = item['question']
    ground_truth = item['answer']
    
    # Retrieve top-K documents using simple FAISS search
    retrieved_docs = rag.retrieve(question, top_k=5)
    contexts_str = "\n".join([doc['text'] for doc in retrieved_docs])
    
    # Traditional RAG Prompt - Khuyến khích trả lời và ép tiếng Việt
    prompt = f"""Bạn là một trợ lý AI phân tích thông tin. Nhiệm vụ của bạn là trả lời câu hỏi dựa trên các tài liệu tham khảo được cung cấp.

HƯỚNG DẪN:
1. Hãy cố gắng hết sức để tổng hợp câu trả lời từ các đoạn trích dẫn dưới đây.
2. Nếu thông tin không hoàn toàn đầy đủ, hãy cố gắng suy luận từ ngữ cảnh để cung cấp câu trả lời hữu ích nhất.
3. Luôn luôn trả lời bằng tiếng Việt.

[Tài liệu tham khảo]
{contexts_str}

[Câu hỏi]
{question}

Câu trả lời của bạn:"""
    
    # Generate Answer
    answer = llm.generate(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "contexts": [doc['text'] for doc in retrieved_docs],
        "ground_truth": ground_truth
    }

def run_traditional_rag_evaluation(num_samples=None, max_workers=4):
    start_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
    output_dir = f"traditional_rag_eval_{start_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Use SimpleRetriever - it will index from scratch into traditional_rag.index/pkl
    rag = SimpleRetriever(index_file="traditional_rag.index", data_file="traditional_rag.pkl", folder_path="./data/sailor_contexts")
        
    llm = LLM(model="llama3:8b")
    
    print("Loading dataset for evaluation...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    if num_samples is None:
        num_samples = len(ds)
    
    items = list(ds.select(range(min(num_samples, len(ds)))))
    results = []
    
    intermediate_csv = os.path.join(output_dir, "traditional_rag_inference.csv")
    intermediate_json = os.path.join(output_dir, "traditional_rag_inference.json")

    print(f"Starting Traditional RAG Inference with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item_traditional_rag, item, rag, llm): i for i, item in enumerate(items)}
        
        pbar = tqdm(total=len(items), desc="Traditional RAG Inference")
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            
            pd.DataFrame([res]).to_csv(
                intermediate_csv, 
                mode='a', 
                header=not os.path.exists(intermediate_csv), 
                index=False, 
                encoding='utf-8-sig'
            )
            
            with open(intermediate_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            pbar.update(1)
        pbar.close()
    
    print(f"Inference complete. Results saved to {output_dir}")

    df = pd.DataFrame(results)
    
    print("Starting Ragas Evaluation...")
    eval_llm = ChatOllama(
        model="llama3:8b", 
        base_url="http://localhost:11434",
        format="json",
        temperature=0
    )
    eval_embeddings = SentenceTransformerEmbeddings(model_name="allenai/specter2_base")
    
    from datasets import Dataset
    if "answer" in df.columns:
        df["response"] = df["answer"]
    ragas_ds = Dataset.from_pandas(df)
    
    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False
    )
    
    print("\nTraditional RAG Evaluation Results:")
    print(result)
    
    output_csv = os.path.join(output_dir, "traditional_rag_final_results.csv")
    result.to_pandas().to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Final metrics saved to {output_csv}")

if __name__ == "__main__":
    run_traditional_rag_evaluation(num_samples=None, max_workers=4)
