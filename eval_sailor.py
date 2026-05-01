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
    
    # Step 1: Retrieval
    retrieved_docs, summary = rag.retrieve(question)
    contexts_str = str([doc['text'] for doc in retrieved_docs])
    
    # Step 2: Multi-Agent Flow
    draft_answer = agent_rag.draft(question, summary, contexts_str)
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
        "draft_answer": draft_answer,
        "critique_1": critique_1,
        "critique_2": critique_2,
        "final_fusion": fusion_output,
        "contexts": [doc['text'] for doc in retrieved_docs],
        "ground_truth": ground_truth
    }

def run_evaluation(num_samples=None, max_workers=4):
    setproctitle.setproctitle("RAG-Eval: Initializing")
    
    rag = Retriever(index_file="sailor.index", data_file="sailor.pkl")
    if not rag.load():
        print("Indexing 4.6k Sailor context files...")
        rag.sync("./data/sailor_contexts")
    
    # Agents will be created per thread or shared (since they are stateless wrappers)
    agents = (RagAgent(), Agent1(), Agent2(), FusionAgent())
    
    print("Loading dataset for evaluation...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    if num_samples is None:
        num_samples = len(ds)
    
    items = list(ds.select(range(min(num_samples, len(ds)))))
    results = []
    
    print(f"Starting Batch Inference with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_item, item, rag, agents): i for i, item in enumerate(items)}
        
        pbar = tqdm(total=len(items), desc="Multi-Agent Benchmark")
        completed = 0
        for future in as_completed(futures):
            results.append(future.result())
            completed += 1
            pbar.update(1)
            setproctitle.setproctitle(f"RAG-Eval: {completed}/{len(items)} processed")
        pbar.close()
    
    df = pd.DataFrame(results)
    
    print("Starting Ragas Evaluation...")
    setproctitle.setproctitle("RAG-Eval: Running Ragas")
    
    eval_llm = ChatOllama(model="llama3:8b", base_url="http://localhost:11434")
    eval_embeddings = SentenceTransformerEmbeddings(model_name="allenai/specter2_base")
    
    from datasets import Dataset
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
    print("Results saved to eval_results_sailor.csv")
    setproctitle.setproctitle("RAG-Eval: Finished")

if __name__ == "__main__":
    # You can adjust max_workers based on your CPU/GPU capacity
    run_evaluation(num_samples=None, max_workers=4)
