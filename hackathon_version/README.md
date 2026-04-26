# 🚀 Multi-Agent RAG System with MinerU Integration

A state-of-the-art Retrieval-Augmented Generation (RAG) pipeline designed for Hackathon 2026. This system features autonomous PDF parsing via a distributed MinerU setup and a multi-agent logic layer for high-precision knowledge synthesis.

---

## 🏗 System Architecture & Workflow

The architecture is partitioned into three decoupled phases to allow for scalability and robust error handling.

### 🔍 Phase 1: Distributed PDF Preprocessing
The **MinerU Middleman** (`mineru_trigger.py`) acts as a bridge to overcome local compute limitations. 
- **Endpoint**: `GET http://localhost:9999/process`
- **Mechanism**: 
  1. Scans the local `./pdf` folder for new documents.
  2. For each document, it triggers a `mineru` CLI command.
  3. The CLI is configured with `--api-url` to point to a high-performance GPU instance (tunneled via `ngrok`).
  4. Extracted files (Markdown, Images, JSON) are saved locally into `./output`.

### 📂 Phase 2: Knowledge Base Construction (ETL)
This phase handles the **Extraction, Transformation, and Loading** of unstructured data into the RAG environment.
- **Goal**: Isolate and verify clean Markdown content.
- **Procedure**:
  - Recursive search: `find ./output -name "*.md"`
  - Migration: `mv ./output/**/*.md ./data_preprocessed/`
  - Integration: `loader.py` logic handles the chunking and initial cleaning of these files.

### 🧠 Phase 3: Multi-Agent RAG Runtime
The core intelligence layer powered by `main.py`.
- **Server**: FastAPI on `http://localhost:8000`
- **Retriever Instance**: Initializes a local vector index from `./data_preprocessed`.
- **Multi-Agent Orchestrator**: 
  - **Drafting Agent**: Generates initial answers based on retrieved context.
  - **Auditor Agent**: Checks for hallucinations and accuracy.
  - **Fusion Agent**: Merges all perspectives into a final, professional response.

---

## 💻 Frontend Interface
The system includes a high-fidelity **Dashboard** (`UI.html`):
- **Aether Theme**: Glassmorphism UI with animated backgrounds.
- **Real-time Streaming**: Visualizes the reasoning steps of each agent.
- **Analytics Tab**: Deep-dive into retrieval performance and confidence scores.

---

## 🛠 Setup & Installation

### 1. Environment Setup
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Dependencies
```bash
pip install -r requirements.txt
pip install fastapi uvicorn requests python-multipart
```

### 3. Directory Structure
```text
├── pdf/                 # Input PDFs
├── output/              # Raw OCR/Parsing output
├── data_preprocessed/   # Cleaned MD files for RAG
├── main.py              # Core RAG API
├── mineru_trigger.py    # PDF Processing Client
├── loader.py            # Data loading utilities
└── UI.html              # Frontend Dashboard
```

---

## 🚦 Operational Guide

| Step | Command | Description |
| :--- | :--- | :--- |
| **Start Proxy** | `python mineru_trigger.py` | Runs the middleman on Port 9999 |
| **Trigger OCR** | `curl http://localhost:9999/process` | Batch converts PDF to MD |
| **Start RAG** | `python main.py` | Launches AI Server on Port 8000 |
| **View UI** | `Open UI.html in Browser` | Access the human interface |

---

## 📦 Tech Stack
- **Parsing**: MinerU (Layout-aware)
- **Frameworks**: FastAPI, Uvicorn
- **AI Logic**: Multi-Agent RAG with Context Streaming
- **Networking**: Ngrok for Distributed Workers

---
*Created for Advanced Agentic Coding - Hackathon 2026*
