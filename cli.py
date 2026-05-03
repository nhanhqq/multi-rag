import argparse
import sys
from core.pdf_parser import parse_pdf_to_text
from core.retriever import Retriever
from core.llm_ollama import OllamaLLM

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    
    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("pdf_path")
    
    query_parser = subparsers.add_parser("query")
    query_parser.add_argument("text")
    
    args = parser.parse_args()
    
    if args.command == "ingest":
        text = parse_pdf_to_text(args.pdf_path)
        retriever = Retriever()
        retriever.add_texts([text], [args.pdf_path])
        print(f"Ingested {args.pdf_path}")
        
    elif args.command == "query":
        retriever = Retriever()
        results = retriever.search(args.text)
        context = "\n".join([r["text"] for r in results])
        llm = OllamaLLM()
        prompt = f"Context: {context}\nQuestion: {args.text}"
        response = llm.generate(prompt)
        print(response)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
