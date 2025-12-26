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

# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
# import torch # type: ignore


# class LLMReasoner:
#     def __init__(
#         self,
#         model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         max_tokens: int = 128,
#     ):
#         self.model_name = model_name
#         self.max_tokens = max_tokens

#         if torch.backends.mps.is_available():
#             self.device = "mps"
#             self.device_index = 0
#             dtype = torch.float32
#         elif torch.cuda.is_available():
#             self.device = "cuda"
#             self.device_index = 0
#             dtype = torch.float16
#         else:
#             self.device = "cpu"
#             self.device_index = -1
#             dtype = torch.float32

#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

#         self.model = AutoModelForCausalLM.from_pretrained(
#             self.model_name,
#             torch_dtype=dtype,
#             low_cpu_mem_usage=True
#         ).to(self.device)

#         self.generator = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device=self.device_index,
#         )

#     def generate_answer(self, query: str, context: str) -> str:
#         if not context or not context.strip():
#             return "I don’t have enough information to answer this question."

#         prompt = (
#             "You are a customer support AI assistant.\n"
#             "Answer ONLY using the provided context.\n"
#             "If the answer is not in the context, say you don't know.\n\n"
#             "Context:\n"
#             f"{context}\n\n"
#             "Question:\n"
#             f"{query}\n\n"
#             "Answer:"
#         )

#         output = self.generator(
#             prompt,
#             max_new_tokens=self.max_tokens,
#             do_sample=False,
#             early_stopping=True,
#             pad_token_id=self.tokenizer.eos_token_id,
#         )

#         text = output[0]["generated_text"]

#         if "Answer:" in text:
#             text = text.split("Answer:", 1)[-1]

#         return text.strip()


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_core.runnables import Runnable   # type: ignore



class LLMReasoner(Runnable):
    def __init__(
        self,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens: int = 128,
        max_context_tokens: int = 2048,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens

        if torch.backends.mps.is_available():
            self.device = 0
            dtype = torch.float16
        else:
            self.device = -1
            dtype = torch.float32

        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        hf_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

        self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "<|system|>\n"
                "You are a customer support AI assistant.\n"
                "Answer ONLY using the provided context.\n"
                "If the answer is not in the context, say you don't know.\n"
                "<|user|>\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n"
                "<|assistant|>\n"
            ),
        )

        self.chain = self.prompt | self.llm

    def invoke(self, inputs: dict) -> str:
        context = inputs.get("context", "").strip()
        question = inputs.get("query", "").strip()

        if not context:
            return "I don’t have enough information to answer this question."

        return self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        ).strip()
