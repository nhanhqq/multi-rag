import os
import json
import pandas as pd
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm
from llm_ollama import LLM
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class SimpleRetriever:
    def __init__(self, data_file="data.pkl", model_name="allenai/specter2_base"):
        print("Loading embeddings from data.pkl...")
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
            self.chunks = data["chunks"]
            self.embeddings = np.array(data["embeddings"]).astype('float32')
        
        print("Building simple FAISS index...")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)
        
        self.model = SentenceTransformer(model_name)
        
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
    
    # Traditional RAG Prompt
    prompt = f"""Bạn là một hệ thống trả lời câu hỏi dựa trên tài liệu (QA System). Hãy đọc kỹ các tài liệu tham khảo được cung cấp và trả lời câu hỏi của người dùng.

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
    
    # Use SimpleRetriever instead of the complex RAG module
    rag = SimpleRetriever(data_file="data.pkl")
        
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
    # max_workers can be adjusted. 1 for strict sequential, 4 for parallel
    run_traditional_rag_evaluation(num_samples=None, max_workers=1)
