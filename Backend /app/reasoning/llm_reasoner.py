# class LLMReasoner:
#     def __init__(self, llm_client):
#         self.llm = llm_client

#     def generate_answer(self, context, user_query):
#         prompt = f"""
# You are an AI assistant.

# RULES:
# - Use ONLY the provided context
# - Do NOT use external knowledge
# - If context is insufficient, say so clearly
# - Be concise and factual

# CONTEXT:
# {context}

# USER QUESTION:
# {user_query}

# ANSWER:
# """
#         response = self.llm(prompt)
#         return response.strip()
    

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
from typing import Optional


class LLMReasoner:
    """
    LLMReasoner
    -----------
    Responsible ONLY for reasoning and answer generation.
    It does NOT handle embeddings, retrieval, or validation.
    """

    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        if not isinstance(model_name, str):
            raise TypeError(
                f"model_name must be a string, got {type(model_name)}"
            )

        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Device selection
        if torch.backends.mps.is_available():
            self.device = "mps"
            self.device_index = 0
            dtype = torch.float16
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.device_index = 0
            dtype = torch.float16
        else:
            self.device = "cpu"
            self.device_index = -1
            dtype = torch.float32

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)

        # Text generation pipeline
        self.generator = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_index,
        )

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer strictly grounded in retrieved context.

        Rules:
        - Use ONLY the provided context
        - If answer is not found, say you don't know
        - Never hallucinate

        Args:
            query (str): User question
            context (str): Retrieved context from vector store

        Returns:
            str: Final clean answer
        """

        if not context or not context.strip():
            return (
                "I don’t have enough information to answer this question "
                "based on the available data."
            )

        prompt = (
            "You are a professional customer support AI assistant.\n\n"
            "STRICT RULES:\n"
            "- Answer ONLY using the provided context\n"
            "- If the answer is not in the context, say: \"I don't know\"\n"
            "- Do NOT add external knowledge\n"
            "- Be concise, polite, and accurate\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Customer Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        output = self.generator(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated_text = output[0]["generated_text"]

        # Remove prompt leakage safely
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:", 1)[-1]

        return generated_text.strip()
