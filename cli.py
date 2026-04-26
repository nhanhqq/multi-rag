"""
cli.py — Simple CLI for the new RAG core.

Usage:
    python cli.py index --pdf_dir ./pdf
    python cli.py ask "What is the main topic?"
"""

import argparse
from retriever import Retriever


def main():
    p = argparse.ArgumentParser(description="RAG CLI (Ollama + MinerU)")
    sub = p.add_subparsers(dest="cmd")

    idx_p = sub.add_parser("index", help="Index PDFs")
    idx_p.add_argument("--pdf_dir", default="./pdf")
    idx_p.add_argument("--output_dir", default="./output")

    ask_p = sub.add_parser("ask", help="Ask a question")
    ask_p.add_argument("query")
    ask_p.add_argument("--top_k", type=int, default=5)
    ask_p.add_argument("--model", default="llama3")

    args = p.parse_args()

    if args.cmd == "index":
        r = Retriever()
        r.load()
        r.index_folder(args.pdf_dir, args.output_dir)
        print(f"Indexed {len(r.chunks)} chunks.")

    elif args.cmd == "ask":
        r = Retriever(model=args.model)
        if not r.load():
            print("No index found. Run: python cli.py index --pdf_dir ./pdf")
            return
        result = r.answer(args.query, top_k=args.top_k)
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nSources: {result['sources']}")

    else:
        p.print_help()


if __name__ == "__main__":
    main()
