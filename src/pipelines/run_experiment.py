import os
import yaml
from pprint import pprint

# Internal imports from our structured architecture
from src.ingestion.document_processor import DocumentIngestor
from src.embeddings.vector_store import EmbeddingManager
from src.retriever.hybrid_search import HybridRetriever
from src.agents.graph import KnowledgeAssistantGraph
from src.evaluation.ragas_eval import evaluate_rag_pipeline
from src.utils.mlflow_logger import log_experiment_run


def main():
    print("=== Starting Enterprise Knowledge Assistant Pipeline (Phase 1) ===")

    # Define paths
    raw_docs_dir = "data/raw_docs"
    vector_store_path = "data/vector_store"
    config_paths = [
        "configs/llm.yaml",
        "configs/retriever.yaml",
        "configs/agent.yaml"
    ]

    # 1. Ingestion & Chunking
    print("\n[1/5] Running Document Ingestion...")
    ingestor = DocumentIngestor(config_path="configs/retriever.yaml")
    chunks = ingestor.process_directory(raw_docs_dir)
    print(f"Generated {len(chunks)} chunks from source documents.")

    # 2. Embedding & Vector Store Management
    print("\n[2/5] Building/Loading Vector Store...")
    embed_manager = EmbeddingManager(config_path="configs/retriever.yaml")
    vector_store = embed_manager.build_or_load_faiss(chunks, save_path=vector_store_path)

    # 3. Initialize Hybrid Retriever
    print("\n[3/5] Initializing Hybrid Retriever & Cross-Encoder...")
    retriever = HybridRetriever(
        vector_store=vector_store,
        chunks=chunks,
        config_path="configs/retriever.yaml"
    )
    #
    # # 4. Initialize LangGraph Agents
    print("\n[4/5] Initializing LangGraph Orchestrator (Qwen 4-bit T4)...")
    graph_orchestrator = KnowledgeAssistantGraph(retriever=retriever)

    # Test a sample query through the Graph
    test_query = "How do I use structural pattern matching in Python 3.10?"
    print(f"\nExecuting Test Query: '{test_query}'")

    initial_state = {
        "query": test_query,
        "intent": "",
        "documents": [],
        "answer": "",
        "confidence_score": 0.0,
        "iterations": 0
    }

    # Invoke the LangGraph compiled state machine
    final_state = graph_orchestrator.graph.invoke(initial_state)
    print("\n--- Final Agent Output ---")
    print(final_state["answer"])
    print(f"(Confidence Score: {final_state['confidence_score']})")

    # 5. Evaluation & MLflow Logging
    print("\n[5/5] Running Evaluation and Logging to MLflow...")
    # In a real run, you would pass a validation dataset. Using a mock setup here.
    test_dataset = [{"query": test_query,
                     "ground_truth": "Structural pattern matching is implemented using the match and case keywords..."}]

    eval_metrics = evaluate_rag_pipeline(test_dataset, graph_orchestrator)

    log_experiment_run(
        config_dirs=config_paths,
        eval_metrics=eval_metrics,
        run_name="qwen_t4_python310_test"
    )

    print("\n=== Pipeline Execution Complete ===")


if __name__ == "__main__":
    main()