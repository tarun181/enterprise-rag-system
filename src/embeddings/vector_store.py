import os
import yaml
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents import Document


class EmbeddingManager:
    def __init__(self, config_path: str = "configs/retriever.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)["embeddings"]

        # BGE requires specific instruction for retrieval optimization
        model_kwargs = {'device': 'cuda'}
        encode_kwargs = {'normalize_embeddings': True}

        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.config["dense_model"],
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs,
            query_instruction="Represent this sentence for searching relevant passages: "
        )
        self.vector_store = None

    def build_or_load_faiss(self, chunks: List[Document], save_path: str = "data/vector_store"):
        """Builds a new FAISS index or loads an existing one."""
        if os.path.exists(save_path) and os.listdir(save_path):
            print(f"Loading existing FAISS index from {save_path}")
            self.vector_store = FAISS.load_local(
                save_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for local trusted FAISS
            )
        else:
            print("Building new FAISS index...")
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
            self.vector_store.save_local(save_path)

        return self.vector_store