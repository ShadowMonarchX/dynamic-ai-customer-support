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


class LLMReasoner:
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_tokens: int = 128,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.device_index = 0
            dtype = torch.float32
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.device_index = 0
            dtype = torch.float16
        else:
            self.device = "cpu"
            self.device_index = -1
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device_index,
        )

    def generate_answer(self, query: str, context: str) -> str:
        if not context or not context.strip():
            return "I don’t have enough information to answer this question."

        prompt = (
            "You are a customer support AI assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not in the context, say you don't know.\n\n"
            "Context:\n"
            f"{context}\n\n"
            "Question:\n"
            f"{query}\n\n"
            "Answer:"
        )

        output = self.generator(
            prompt,
            max_new_tokens=self.max_tokens,
            do_sample=False,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        text = output[0]["generated_text"]

        if "Answer:" in text:
            text = text.split("Answer:", 1)[-1]

        return text.strip()
