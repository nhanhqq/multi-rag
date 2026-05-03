# VZeDo: A Cost-Efficient Multi-Agent Framework for Zero-Shot Domain-Specific Retrieval-Augmented Generation

<p align="center">
  <img src="https://img.shields.io/badge/Model-Sailor-blue?style=for-the-badge" alt="Model Sailor">
  <img src="https://img.shields.io/badge/Framework-Multi--Agent-orange?style=for-the-badge" alt="Multi-Agent">
  <img src="https://img.shields.io/badge/Efficiency-Cost--Effective-green?style=for-the-badge" alt="Cost Effective">
</p>

## Abstract
[To be updated]

---

## 💡 Key Highlights

### 💸 Cost-Efficiency & Resource Optimization
- **100% Local Inference**: Zero API costs via **Ollama**.
- **Context Compression**: Smart sentence filtering reduces prompt size by **~60%**.
- **Lean Model Stack**: High-accuracy, low-compute components (`Specter2`, `MiniLM`).
- **Hallucination-Driven Savings**: Evaluator agents prevent wasted compute on false paths.

### 🤖 Multi-Agent Orchestration
- **Reflexion Architecture**: `RagAgent` -> `Evaluators` -> `FusionAgent`.
- **Hybrid Retrieval**: Fusion of **FAISS** (dense) and **BM25** (sparse) with Cross-Encoder reranking.

### 🎯 Zero-Shot Domain Adaptation
- Ready for specialized domains (Legal, SciFact, Medical) without fine-tuning.

---

## 📊 Evaluation
Supports **RAGAS** metrics and automated benchmarking for domain-specific datasets.

---

## 💻 Quick Start
```bash
pip install -r requirements.txt
# Run locally with Ollama (e.g., sailor2:8b)
```

---
**Maintained by**: nhanhq | **Core Focus**: Cost-Efficient Multi-Agent RAG.
