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

from agents import RagAgent, Agent1, Agent2, FusionAgent

def run_evaluation(num_samples=None):
    rag = Retriever(index_file="sailor.index", data_file="sailor.pkl")
    if not rag.load():
        print("Indexing 4.6k Sailor context files...")
        rag.sync("./data/sailor_contexts")
    
    # Initialize Multi-Agent Pipeline
    agent_rag = RagAgent()
    agent_1 = Agent1()
    agent_2 = Agent2()
    agent_fusion = FusionAgent()
    
    print("Loading dataset for evaluation...")
    ds = load_dataset("sailor2/Vietnamese_RAG", "BKAI_RAG", split="train")
    if num_samples is None:
        num_samples = len(ds)
    
    results = []
    
    for i, item in enumerate(tqdm(ds.select(range(min(num_samples, len(ds)))), desc="Multi-Agent Benchmark")):
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
        
        results.append({
            "question": question,
            "draft_answer": draft_answer,
            "critique_1": critique_1,
            "critique_2": critique_2,
            "final_fusion": fusion_output,
            "contexts": [doc['text'] for doc in retrieved_docs],
            "ground_truth": ground_truth
        })
    
    df = pd.DataFrame(results)
    
    from langchain_community.chat_models import ChatOllama
    eval_llm = ChatOllama(model="qwen3.5:0.8b", base_url="http://localhost:11434")
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
