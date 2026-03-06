# Enterprise Knowledge Assistant (Phase 1)

This repository contains the core intelligence layer of a multi-agent Retrieval-Augmented Generation (RAG) system. It is specifically optimized to run an open-source model (Qwen 3.5 7B) on a single NVIDIA T4 GPU (16GB VRAM) using 4-bit NF4 quantization.

## System Architecture
The system uses **LangGraph** to route queries intelligently rather than relying on a static chain.
1. **Router Agent**: Directs traffic.
2. **Retrieval Agent**: Hybrid FAISS (Dense) + BM25 (Sparse) + Cross-Encoder Reranking.
3. **Summarization Agent**: Synthesizes output.
4. **Critic Agent**: Validates output to prevent hallucinations.

## Setup Instructions
1. Install requirements: `pip install -r requirements.txt` (including `transformers`, `bitsandbytes`, `langgraph`, `faiss-cpu`, `mlflow`).
2. Place raw HTML/Markdown files in `data/raw_docs/`.
3. Adjust `configs/` YAML files as needed.
4. Run Ingestion: `python src/pipelines/ingest_data.py`
5. Run Evaluation/MLflow: `python src/pipelines/run_experiment.py`
6. View logs: `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`