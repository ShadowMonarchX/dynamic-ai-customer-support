
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class LocalLLM:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.device == "mps" else -1,
        )

        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer using retrieved context.

        IMPORTANT:
        - The model MUST answer only from context
        - If context is empty → safe fallback response

        Args:
            query: User question
            context: Retrieved text from vector DB

        Returns:
            Clean AI-generated response
        """

        if not context.strip():
            return (
                "I don’t have this information yet. "
                "Please contact customer support for further assistance."
            )

        prompt = f"""
You are a customer support AI assistant.

Rules:
- Answer ONLY using the provided context
- If the answer is not in the context, say you don't know
- Be clear, polite, and professional

Context:
{context}

Customer Question:
{query}

Answer:
"""

        output = self.generator(
            prompt,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        response = output[0]["generated_text"]

        # Post-processing: remove prompt leakage
        if "Answer:" in response:
            response = response.split("Answer:")[-1]

        return response.strip()
