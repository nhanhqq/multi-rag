import json
import os

def check_retrieval_recall():
    from core.RAG import Retriever
    rag = Retriever()
    if not rag.load():
        print("Could not load RAG index")
        return
        
    with open("scifact_data/claims_dev.jsonl", "r") as f:
        claims = [json.loads(l) for l in f]
        
    correct_retrieval = 0
    total = 0
    
    print("Evaluating Retriever Recall on first 50 claims...")
    for claim_obj in claims[:50]:
        q = claim_obj["claim"]
        gold_docs = [str(x) for x in claim_obj.get("cited_doc_ids", [])]
        if not gold_docs:
            continue
            
        chunks, _ = rag.retrieve(q, top_k=5)
        retrieved_docs = [c['source'].replace('.md', '') for c in chunks]
        
        hit = False
        for g in gold_docs:
            if g in retrieved_docs:
                hit = True
                break
                
        if hit:
            correct_retrieval += 1
        else:
            print(f"Missed: {q[:50]}... | Gold: {gold_docs} | Got: {retrieved_docs}")
        total += 1
        
    print(f"\nRetrieval Recall@5: {correct_retrieval/total:.2%} ({correct_retrieval}/{total})")

if __name__ == "__main__":
    check_retrieval_recall()
