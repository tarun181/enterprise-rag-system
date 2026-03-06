import yaml
import numpy as np
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from sentence_transformers import CrossEncoder
try:
    from langchain.retrievers import EnsembleRetriever
except ImportError:
    try:
        from langchain_community.retrievers import EnsembleRetriever
    except ImportError:
        from langchain_classic.retrievers import EnsembleRetriever


class HybridRetriever:
    def __init__(self, vector_store, chunks: List[Document], config_path: str = "configs/retriever.yaml"):
        # 1. Load the YAML once
        with open(config_path, "r") as f:
            full_config = yaml.safe_load(f)
            self.config = full_config["search"]
            self.embed_config = full_config["embeddings"]

        # 2. Setup Dense Retriever (FAISS)
        self.dense_retriever = vector_store.as_retriever(
            search_kwargs={"k": self.config["top_k_retrieve"]}
        )

        # 2. Setup Sparse Retriever (BM25) - excellent for code/exact terms
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = self.config["top_k_retrieve"]

        # 3. Combine with Ensemble Retriever
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.dense_retriever],
            weights=[self.config["hybrid_alpha"], 1 - self.config["hybrid_alpha"]]
        )

        # 4. Load Cross-Encoder for Reranking (runs on T4 GPU)
        self.cross_encoder = CrossEncoder(
            self.embed_config["cross_encoder"],
            device="cuda",
            max_length=512
        )

    def retrieve_and_rerank(self, query: str) -> List[Dict[str, Any]]:
        """Executes hybrid search and reranks using a cross-encoder."""
        # Step 1: Broad hybrid retrieval
        initial_docs = self.ensemble_retriever.invoke(query)

        if not initial_docs:
            return []

        # Step 2: Prepare pairs for cross-encoder scoring
        pairs = [[query, doc.page_content] for doc in initial_docs]

        # Step 3: Score and sort
        scores = self.cross_encoder.predict(pairs)

        # Combine docs with scores
        scored_docs = list(zip(initial_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # Step 4: Keep Top-K final documents and format with citations
        top_docs = scored_docs[:self.config["top_k_final"]]

        results = []
        for doc, score in top_docs:
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "rerank_score": float(score)
            })

        return results