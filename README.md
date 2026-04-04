# 🚀 Multi-Agent Hybrid RAG System

An advanced **Retrieval-Augmented Generation (RAG)** pipeline that leverages **Hybrid Search** (Dense + Sparse), **Cross-Encoder Reranking**, and an innovative **Iterative Multi-Agent Synthesis** workflow to deliver meticulously fact-checked and professionally toned answers.

---

## 🌟 Key Features

1. **Hybrid Retrieval Pipeline 🔍**
   - **Dense Search** using `SentenceTransformers` (`all-MiniLM-L6-v2`) via `FAISS` for pure semantic matching.
   - **Sparse Search** using lexical `BM25Okapi` to capture exact keyword matches.
2. **Deep Reranking 🧠**
   - Utilizes `Cross-Encoder` (`ms-marco-MiniLM-L-6-v2`) to intelligently evaluate the contextual overlap between the query and candidates.
   - Applies **MMR (Maximal Marginal Relevance)** to ensure the final context is highly relevant yet diverse.
3. **Multi-Agent Synthesis (Powered by Groq) 🤖**
   Rather than handing data to a single LLM blindly, the system orchestrates 4 independent agents:
   - **RagAgent (Draft Generator)**: Extrapolates a raw draft directly from the context.
   - **Agent 1 (Grounding Auditor)**: Aggressively checks the Draft against the context to eliminate hallucinations and factual inaccuracies.
   - **Agent 2 (Tone & Logic Judge)**: Critiques the professional tone and logical coherence of the Draft.
   - **FusionAgent (Executive Synthesizer)**: Absorbs all critiques, utilizing external generalized knowledge strictly as a fallback, to synthesize a flawless, citation-rich final response.

---

## 🛠 Project Structure

```bash
├── pdf/                       # Drop your raw PDF/TXT files here
├── system_promt/              # Strict Instruction files for LLM Agents
│   ├── rag.txt                # Draft Generation Prompt
│   ├── agent1.txt             # Grounding Auditor Prompt
│   ├── agent2.txt             # Tone Judge Prompt
│   └── agent3.txt             # Final Fusion Synthesizer Prompt
├── RAG.py                     # The core Hybrid Retrieval Engine (FAISS + BM25)
├── llm.py                     # Groq LLM API Wrapper (cleans context & responses)
├── agents.py                  # Agent classes loading specific execution prompts
├── init.py                    # Sync and Indexing manager
├── main.py                    # Main Orchestrator & CLI execution loop
└── pipeline_diagram.html      # Highly detailed UI diagram of the system pipeline
```

---

## ⚡ Getting Started

### 1. Prerequisites
You need a [Groq API](https://console.groq.com) account. In your root folder, create a `.env` file containing 4 distinct keys (this circumvents strict rate limits):
```env
GROQ_API_KEY1=gsk_yourkey_here
GROQ_API_KEY2=gsk_yourkey_here
GROQ_API_KEY3=gsk_yourkey_here
GROQ_API_KEY4=gsk_yourkey_here
```

### 2. Installations
Ensure you have the required ML and OS libraries installed:
```bash
pip install -r requirements.txt
# (Includes PyMuPDF (fitz), faiss-cpu, sentence-transformers, rank_bm25, groq, python-dotenv, nltk)
```

### 3. Usage
Simply execute the main orchestration file:
```bash
python main.py
```
> **Note**: On the first execution, `init.py` will read all files in your `./pdf` directory, extract them, and build the persistent `data.index` and `data.pkl` representations. 

---

## 📊 System Architecture

Want a visual representation of how this works? Open the `pipeline_diagram.html` file in your favorite browser to view the entire architectural flow—from Offline Data Indexing to Online Multi-Agent Synthesis—beautifully rendered!
