import torch
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline

logger = logging.getLogger(__name__)


class OptimizedInferenceEngine:
    def __init__(self, model_path: str = "Qwen/Qwen3.5-4B"):
        """
        Initializes the Hugging Face pipeline.
        - Uses 4-bit NF4 quantization to fit the T4 GPU.
        - Uses 'sdpa' (Scaled Dot Product Attention) for FlashAttention-like speedups natively on Windows.
        """
        logger.info(f"Loading {model_path} into memory...")
        import os

        # Point to your local downloaded models directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        local_model_path = os.path.join(project_root, "models", "Qwen3.5-4B")

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)

        # attn_implementation="sdpa" provides massive memory/speed optimizations 
        # without needing custom Linux binaries
        self.model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            quantization_config=bnb_config,
            device_map="auto",
            attn_implementation="sdpa"
        )

        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=1024,
            temperature=0.1,
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False  # Crucial for APIs so it only returns the answer, not the prompt
        )
        logger.info("HF Pipeline engine initialized successfully.")

    def generate(self, prompt: str) -> str:
        """Generates a response using the optimized HF pipeline."""
        outputs = self.pipeline(prompt)
        return outputs[0]["generated_text"].strip()