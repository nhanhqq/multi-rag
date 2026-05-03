import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from core.llm_ollama import LLM
from ragas import evaluate
from ragas.metrics import answer_relevancy
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

def process_item_zero_shot(item, llm):
    question = item['question']
    ground_truth = item['answer']
    
    # Zero-shot Prompt - Ép trả lời tiếng Việt
    prompt = f"Hãy trả lời câu hỏi sau đây bằng tiếng Việt:\nCâu hỏi: {question}\nTrả lời:"
    
    # Generate Answer
    answer = llm.generate(prompt)
    
    return {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
        "contexts": [] # Trống cho Zero-shot
    }

def run_zero_shot_evaluation(num_samples=None, max_workers=4):
    start_time = datetime.now().strftime("%d_%m_%Y_%H_%M")
    output_dir = f"zero_shot_eval_{start_time}"
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading dataset for evaluation...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    if num_samples is None:
        num_samples = len(ds)
    
    items = list(ds.select(range(min(num_samples, len(ds)))))
    results = []
    
    llm = LLM(model="llama3:8b")
    
    intermediate_csv = os.path.join(output_dir, "zero_shot_inference.csv")
    intermediate_json = os.path.join(output_dir, "zero_shot_inference.json")

    print(f"Starting Zero-shot Inference with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item_zero_shot, item, llm): i for i, item in enumerate(items)}
        
        pbar = tqdm(total=len(items), desc="Zero-shot Inference")
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
    
    print("Starting Ragas Evaluation (answer_relevancy only)...")
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
        metrics=[answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False
    )
    
    print("\nZero-shot Evaluation Results:")
    print(result)
    
    output_csv = os.path.join(output_dir, "zero_shot_final_results.csv")
    result.to_pandas().to_csv(output_csv, index=False, encoding='utf-8-sig')
    print(f"Final metrics saved to {output_csv}")

if __name__ == "__main__":
    # max_workers can be adjusted. 1 for strict sequential, 4 for parallel
    run_zero_shot_evaluation(num_samples=None, max_workers=1)
