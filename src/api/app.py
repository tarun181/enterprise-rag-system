from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time
import logging

# Import the optimized inference engine
from src.inference.inference_utils import OptimizedInferenceEngine
# Import the retriever you built in Phase 1
from src.retriever.hybrid_search import HybridRetriever
from src.ingestion.document_processor import DocumentIngestor
from src.embeddings.vector_store import EmbeddingManager

app = FastAPI(title="Enterprise Knowledge Assistant API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Initialize the inference engine
llm_engine = OptimizedInferenceEngine(model_path="Qwen/Qwen3.5-4B")

# 2. Initialize the Real Retriever
try:
    logger.info("Ingesting chunks and loading FAISS Vector Store...")

    # Load the raw text chunks so BM25 can build its keyword index
    ingestor = DocumentIngestor()
    chunks = ingestor.process_directory("data/raw_docs")

    # Connect to your local FAISS database AND load the index
    embedding_manager = EmbeddingManager()
    faiss_index = embedding_manager.build_or_load_faiss(chunks)

    # Wire them both into the Hybrid Retriever!
    hybrid_retriever = HybridRetriever(vector_store=faiss_index, chunks=chunks)
    logger.info("✅ Hybrid Retriever successfully loaded and ready for queries!")

except Exception as e:
    logger.error(f"Failed to load retriever: {e}")
    hybrid_retriever = None


class QueryRequest(BaseModel):
    query: str


class Citation(BaseModel):
    source: str
    section: str
    url: str


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    confidence_score: float
    anomaly_score: float
    processing_time: float


@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):
    start_time = time.time()

    try:
        context = ""
        citations = []
        # 3. Dynamic Retrieval Phase
        if hybrid_retriever:
            # Call your custom reranking method!
            docs = hybrid_retriever.retrieve_and_rerank(request.query)

            # Format the real context and build dynamic citations
            context_parts = []

            # Take the top 3 most relevant chunks
            for i, doc in enumerate(docs[:3]):
                # Notice we use dict lookups ['content'] instead of .page_content
                context_parts.append(f"--- Document {i + 1} ---\n{doc['content']}")

                # Extract the metadata
                citations.append(
                    Citation(
                        source=doc['metadata'].get("source", f"Document {i + 1}"),
                        section=doc['metadata'].get("Section", "General"),
                        url=doc['metadata'].get("url", "#")
                    )
                )
            context = "\n\n".join(context_parts)
        else:
            context = "No knowledge base available. The FAISS index could not be loaded."

        # 4. Prompt Assembly
        formatted_prompt = f"""<|im_start|>system
You are a helpful enterprise assistant. Answer based ONLY on the context provided. If the answer is not in the context, explicitly say "I do not have enough information to answer this."<|im_end|>
<|im_start|>user
Context: {context}
Query: {request.query}<|im_end|>
<|im_start|>assistant
"""

        # 5. Optimized Generation
        generated_answer = llm_engine.generate(formatted_prompt)

        processing_time = time.time() - start_time

        return QueryResponse(
            answer=generated_answer,
            citations=citations,
            confidence_score=0.95,  # We will wire up the real critic score in Phase 3
            anomaly_score=0.01,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error during query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)