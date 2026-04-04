# 🚀 Multi-Agent Hybrid RAG System (with Confidence Scoring)

An advanced **Retrieval-Augmented Generation (RAG)** pipeline that leverages **Hybrid Search** (Dense + Sparse), **Cross-Encoder Reranking**, and an innovative **Iterative Multi-Agent Synthesis** workflow. This system ensures every answer is fact-checked and ends with a **Confidence Score (0-100)** for maximum reliability.

---

## 🌟 Key Features

1. **Intelligent Ingestion with LlamaIndex 📁**
   - Supports multi-format data ingestion (PDF, TXT, MD, DOCX) via `SimpleDirectoryReader`.
   - Semantic chunking using `SentenceSplitter` to maintain context integrity.
2. **Hybrid Retrieval Pipeline 🔍**
   - **Dense Search** using `SentenceTransformers` (`all-MiniLM-L6-v2`) via `FAISS` for pure semantic matching.
   - **Sparse Search** using lexical `BM25Okapi` to capture exact keyword matches.
3. **Deep Reranking & Diversity 🧠**
   - Utilizes `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`) for top-tier precision.
   - Applies **MMR (Maximal Marginal Relevance)** and **Context Compression** to provide token-efficient, hit-dense context.
4. **Multi-Agent Synthesis (Powered by Groq Llama 4) 🤖**
   Orchestrates 4 independent agents for a "trust but verify" workflow:
   - **RagAgent (Draft Generator)**: Builds the initial answer from retrieved context.
   - **Agent 1 (Grounding Auditor)**: Verifies every sentence against the source to prevent hallucinations.
   - **Agent 2 (Logic & Tone Judge)**: Ensures logical flow and professional delivery.
   - **FusionAgent (Executive Synthesizer)**: Blends all critiques using the **Llama 4 Scout 17B** engine and provides a final **Confidence Score**.

---

## 🛠 Project Structure

```bash
├── pdf/                       # Ingestion directory for raw documents
├── system_promt/              # Strict Agent instructions (Prompt engineering)
│   ├── rag.txt                # Draft Generation
│   ├── agent1.txt             # Accuracy Audit
│   ├── agent2.txt             # Logic & Style Review
│   └── agent3.txt             # Final Fusion + Confidence Score
├── RAG.py                     # Hybrid Engine (FAISS + BM25 + Reranking)
├── loader.py                  # LlamaIndex Document Loader & Parser
├── llm.py                     # Groq LLM API Wrapper (cleaning & generation)
├── agents.py                  # Agent definitions & multi-key rotation
├── main.py                    # Main CLI interface & Orchestration loop
└── pipeline_diagram.html      # Detailed Visual Workflow Diagram
```

---

## ⚡ Getting Started

### 1. Prerequisites
Get your [Groq API](https://console.groq.com) keys. Create a `.env` file with 4 keys to optimize throughput:
```env
GROQ_API_KEY1=gsk_...
GROQ_API_KEY2=gsk_...
GROQ_API_KEY3=gsk_...
GROQ_API_KEY4=gsk_...
```

### 2. Quick Install
```bash
pip install faiss-cpu sentence-transformers rank_bm25 groq python-dotenv nltk llama-index llama-index-llms-groq
```

### 3. Usage
```bash
python main.py
```

---

## 📊 System Architecture

For a deep dive into the technical flow, open `pipeline_diagram.html` in your browser. It visualizes the entire process from Offline Indexing to the Multi-Agent Confidence Scoring loop.
