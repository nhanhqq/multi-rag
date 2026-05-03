import os
import argparse
import json
import requests
import re
from tqdm import tqdm
from fpdf import FPDF
from pathlib import Path

from core.RAG import Retriever
from core.agents import RagAgent, Agent1, Agent2, FusionAgent

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=30, help="Number of chunks to retrieve")
    args = parser.parse_args()
    
    corpus_file, claims_file = "scifact_data/corpus.jsonl", "scifact_data/claims_dev.jsonl"
    pdf_dir, md_dir = "scifact_pdfs", "scifact_mds"
    process_corpus(corpus_file, pdf_dir, md_dir)
    
    rag = Retriever()
    if not rag.load():
        md_files = []
        for fname in os.listdir(md_dir):
            if fname.endswith(".md"):
                with open(os.path.join(md_dir, fname), "r", encoding="utf-8") as f:
                    md_files.append({"source": fname, "text": f.read()})
        if md_files:
            rag.sync(md_files)
            
    rag_agent = RagAgent()
    agent1 = Agent1()
    agent2 = Agent2()
    fusion_agent = FusionAgent()
    
    results = []
    with open(claims_file, "r") as f:
        claims = [json.loads(l) for l in f.readlines()]
        
    output_dir = "scifact_output"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"results_topk_{args.top_k}.json")
    
    for claim_obj in tqdm(claims, desc="Evaluating"):
        q = claim_obj["claim"]
        chunks, summary = rag.retrieve(q, top_k=30)
        c_text = " ".join([f"{r['source']} {r['text']}" for r in chunks])
        
        draft = rag_agent.draft(q, summary, c_text)
        e1 = agent1.evaluate(q, c_text, draft)
        e2 = agent2.evaluate(q, summary, draft)
        final = fusion_agent.fuse(q, draft, e1, e2, summary, c_text)
        
        try:
            clean_final = final.strip()
            if clean_final.startswith("```json"): clean_final = clean_final[7:]
            elif clean_final.startswith("```"): clean_final = clean_final[3:]
            if clean_final.endswith("```"): clean_final = clean_final[:-3]
            res_dict = json.loads(clean_final.strip())
            parsed_label = str(res_dict.get("label", "NOT_ENOUGH_INFO")).upper()
            if "SUPPORT" in parsed_label: label = "SUPPORT"
            elif "REFUTE" in parsed_label or "CONTRADICT" in parsed_label: label = "REFUTES"
            else: label = "NOT_ENOUGH_INFO"
        except Exception:
            label = "NOT_ENOUGH_INFO"
        
        results.append({
            "id": claim_obj["id"],
            "claim": q,
            "predicted": label,
            "gold": claim_obj.get("evidence_label", "UNKNOWN"),
            "final_response": final
        })
        with open(output_file, "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
