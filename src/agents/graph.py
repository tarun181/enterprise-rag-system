import operator
import os
from typing import TypedDict, Annotated, Sequence, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_huggingface import HuggingFacePipeline
from src.retriever.hybrid_search import HybridRetriever
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch


# Define Graph State
class AgentState(TypedDict):
    query: str
    intent: str
    documents: list[Dict[str, Any]]
    answer: str
    confidence_score: float
    iterations: int


class KnowledgeAssistantGraph:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.llm = self._load_qwen_4bit()  # Utility to load Qwen 3.5 7B via HF pipeline
        self.graph = self._build_graph()

    def _load_qwen_4bit(self):
        """Loads Qwen LLM optimized for 16GB VRAM T4 using 4-bit NF4."""
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        local_model_path = os.path.join(project_root, "models", "Qwen3.5-4B")

        print(f"Loading LLM from local path: {local_model_path}")

        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map="auto"
        )

        return HuggingFacePipeline.from_model_id(
            model_id=local_model_path,
            task="text-generation",
            pipeline_kwargs={"max_new_tokens": 1024}
        )

    # --- AGENT NODES ---

    def router_agent(self, state: AgentState):
        """Classifies query intent."""
        # In production, use an LLM call here with a strict prompt.
        # Mocking logic for brevity:
        intent = "informational"
        if "list all" in state["query"].lower(): intent = "structured"
        return {"intent": intent, "iterations": state.get("iterations", 0) + 1}

    def retrieval_agent(self, state: AgentState):
        """Executes RAG pipeline."""
        docs = self.retriever.retrieve_and_rerank(state["query"])
        return {"documents": docs}

    def tool_agent(self, state: AgentState):
        """Queries structured metadata directly."""
        # E.g., querying a SQLite DB or purely metadata matching
        return {"answer": "Structured tool output: List of docs...", "confidence_score": 1.0}

    def summarization_agent(self, state: AgentState):
        """Generates answer using retrieved context."""
        context = "\n\n".join([f"Source: {d['metadata']['url']}\n{d['content']}" for d in state["documents"]])
        prompt = f"Answer the query based ONLY on the context. Cite sources.\nQuery:{state['query']}\nContext:{context}"
        answer = self.llm.invoke(prompt)
        return {"answer": answer}

    def critic_agent(self, state: AgentState):
        """Checks for hallucinations."""
        # Ask LLM to score the answer against the context.
        # Mocking output:
        score = 0.95  # High confidence
        return {"confidence_score": score}

    # --- EDGE LOGIC ---

    def route_query(self, state: AgentState):
        if state["intent"] == "structured": return "tool_agent"
        return "retrieval_agent"

    def check_confidence(self, state: AgentState):
        if state["confidence_score"] < 0.7 and state["iterations"] < 3:
            return "retrieval_agent"  # Loop back and try retrieving differently
        return END

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        # Add Nodes
        workflow.add_node("router_agent", self.router_agent)
        workflow.add_node("retrieval_agent", self.retrieval_agent)
        workflow.add_node("tool_agent", self.tool_agent)
        workflow.add_node("summarization_agent", self.summarization_agent)
        workflow.add_node("critic_agent", self.critic_agent)

        # Set Edges
        workflow.set_entry_point("router_agent")
        workflow.add_conditional_edges("router_agent", self.route_query)
        workflow.add_edge("retrieval_agent", "summarization_agent")
        workflow.add_edge("summarization_agent", "critic_agent")
        workflow.add_conditional_edges("critic_agent", self.check_confidence)
        workflow.add_edge("tool_agent", END)

        return workflow.compile()