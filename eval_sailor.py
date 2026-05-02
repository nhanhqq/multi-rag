import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from RAG import Retriever
from llm_ollama import LLM
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from agents import RagAgent, Agent1, Agent2, FusionAgent
from concurrent.futures import ThreadPoolExecutor, as_completed
import setproctitle

def process_item(item, rag, agents):
    agent_rag, agent_1, agent_2, agent_fusion = agents
    question = item['question']
    ground_truth = item['answer']
    
    retrieved_docs, summary = rag.retrieve(question)
    contexts_str = str([doc['text'] for doc in retrieved_docs])
    
    draft_answer = agent_rag.draft(question, summary, contexts_str)
    
    if "không đủ" in draft_answer.lower() or "not enough" in draft_answer.lower():
        print(f"\n[CẢNH BÁO] RagAgent từ chối trả lời câu hỏi: {question}")

    critique_1 = agent_1.evaluate(question, contexts_str, draft_answer)
    critique_2 = agent_2.evaluate(question, summary, draft_answer)
    
    fusion_output = agent_fusion.fuse(
        question, 
        draft_answer, 
        critique_1, 
        critique_2, 
        summary, 
        contexts_str
    )
    
    return {
        "question": question,
        "answer": fusion_output,
        "contexts": [doc['text'] for doc in retrieved_docs],
        "ground_truth": ground_truth,
        "draft_answer": draft_answer,
        "critique_1": critique_1,
        "critique_2": critique_2
    }

def run_evaluation(num_samples=None, max_workers=4):
    setproctitle.setproctitle("RAG-Eval: Initializing")
    
    rag = Retriever(index_file="sailor.index", data_file="sailor.pkl")
    if not rag.load():
        print("Indexing 4.6k Sailor context files...")
        rag.sync("./data/sailor_contexts")
    
    agents = (RagAgent(), Agent1(), Agent2(), FusionAgent())
    
    print("Loading dataset for evaluation...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    if num_samples is None:
        num_samples = len(ds)
    
    items = list(ds.select(range(min(num_samples, len(ds)))))
    results = []
    
    intermediate_csv = "eval_results_sailor_inference.csv"
    intermediate_json = "eval_results_sailor_inference.json"
    
    if os.path.exists(intermediate_csv):
        os.remove(intermediate_csv)
    if os.path.exists(intermediate_json):
        os.remove(intermediate_json)

    print(f"Starting Batch Inference with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item, rag, agents): i for i, item in enumerate(items)}
        
        pbar = tqdm(total=len(items), desc="Multi-Agent Benchmark")
        completed = 0
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
            
            completed += 1
            pbar.update(1)
            setproctitle.setproctitle(f"RAG-Eval: {completed}/{len(items)} processed")
        pbar.close()
    
    print(f"Inference complete. Results saved to {intermediate_csv} and {intermediate_json}")

    df = pd.DataFrame(results)
    
    print("Starting Ragas Evaluation...")
    setproctitle.setproctitle("RAG-Eval: Running Ragas")
    
    eval_llm = ChatOllama(
        model="llama3:8b", 
        base_url="http://localhost:11434",
        format="json",
        temperature=0
    )
    eval_embeddings = SentenceTransformerEmbeddings(model_name="allenai/specter2_base")
    
    from datasets import Dataset
    ragas_ds = Dataset.from_pandas(df)
    
    if "answer" in df.columns:
        df["response"] = df["answer"]
        ragas_ds = Dataset.from_pandas(df)

    result = evaluate(
        ragas_ds,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=eval_llm,
        embeddings=eval_embeddings
    )
    
    print("\nEvaluation Results:")
    print(result)
    
    result.to_pandas().to_csv("eval_results_sailor.csv", index=False)
    print("Final results saved to eval_results_sailor.csv")
    setproctitle.setproctitle("RAG-Eval: Finished")

if __name__ == "__main__":
    run_evaluation(num_samples=None, max_workers=1)
