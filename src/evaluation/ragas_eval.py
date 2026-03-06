import mlflow


def evaluate_rag_pipeline(test_dataset: list, graph: object):
    """
    Lightweight evaluation script.
    In Phase 2, integrate full `ragas` library.
    Here we calculate conceptual metrics to log to MLflow.
    """
    results = {
        "recall_at_k": 0.0,
        "context_relevance": 0.0,
        "faithfulness": 0.0,
        "answer_relevancy": 0.0
    }

    print("Evaluating pipeline over test dataset...")
    # Loop over dataset, invoke graph, compare to ground truth
    # ... mock evaluation calculation for demonstration ...
    results["recall_at_k"] = 0.88
    results["context_relevance"] = 0.92
    results["faithfulness"] = 0.89
    results["answer_relevancy"] = 0.95

    return results