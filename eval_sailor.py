import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from RAG import Retriever
from llm_ollama import LLM
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings

def run_evaluation(num_samples=10):
    rag = Retriever()
    if not rag.load():
        print("Indexing data...")
        rag.sync("./data/sailor_contexts")
    
    llm = LLM()
    
    print("Loading dataset for evaluation...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    
    results = []
    
    prompt_path = "system_promt/rag_vn.txt"
    if not os.path.exists(prompt_path):
        prompt_path = "system_promt/rag.txt"
        
    with open(prompt_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    for i, item in enumerate(tqdm(ds.select(range(min(num_samples, len(ds)))), desc="Inference")):
        question = item['question']
        ground_truth = item['answer']
        
        retrieved_docs, summary = rag.retrieve(question)
        contexts = [doc['text'] for doc in retrieved_docs]
        
        prompt = f"User Query: {question}\nConsolidated Summary: {summary}\nRetrieved Chunks: {contexts}"
        answer = llm.generate(prompt, system_prompt=system_prompt)
        
        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })
    
    df = pd.DataFrame(results)
    
    from langchain_community.chat_models import ChatOllama
    eval_llm = ChatOllama(model="deepseek-v4-flash:cloud", base_url="http://localhost:11434")
    eval_embeddings = SentenceTransformerEmbeddings(model_name="allenai/specter2_base")
    
    from datasets import Dataset
    ragas_ds = Dataset.from_pandas(df)
    
    print("Running Ragas evaluation...")
    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\nEvaluation Results:")
    print(result)
    
    result.to_pandas().to_csv("eval_results_sailor.csv", index=False)
    print("Results saved to eval_results_sailor.csv")

if __name__ == "__main__":
    run_evaluation()
