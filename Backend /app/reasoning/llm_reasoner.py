# import torch #type: ignore
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline #type: ignore
# from langchain_huggingface import HuggingFacePipeline #type: ignore
# from langchain_core.prompts import PromptTemplate #type: ignore
# from langchain_core.runnables import Runnable #type: ignore

# class LLMReasoner(Runnable):
#     def __init__(
#         self,
#         model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#         max_new_tokens: int = 256,
#     ):
#         self.model_name = model_name
#         self.max_new_tokens = max_new_tokens

#         if torch.backends.mps.is_available():
#             self.device = 0
#             dtype = torch.float16
#         elif torch.cuda.is_available():
#             self.device = 0
#             dtype = torch.float16
#         else:
#             self.device = -1
#             dtype = torch.float32

#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
#         self.model = AutoModelForCausalLM.from_pretrained(
#             model_name,
#             torch_dtype=dtype,
#             low_cpu_mem_usage=True
#         )

#         hf_pipeline = pipeline(
#             "text-generation",
#             model=self.model,
#             tokenizer=self.tokenizer,
#             device=self.device,
#             max_new_tokens=self.max_new_tokens,
#             do_sample=False,
#             pad_token_id=self.tokenizer.eos_token_id
#         )

#         self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

#         self.prompt = PromptTemplate(
#             input_variables=["context", "question"],
#             template=(
#                 "You are a customer support AI assistant.\n"
#                 "Answer ONLY using the provided context.\n"
#                 "If the answer is not in the context, say you don't know.\n\n"
#                 "Context:\n{context}\n\n"
#                 "Question:\n{question}\n\n"
#                 "Answer:"
#             )
#         )

#         self.chain = self.prompt | self.llm

#     def invoke(self, inputs: dict) -> str:
#         context = inputs.get("context", "").strip()
#         question = inputs.get("query", "").strip()

#         if not context:
#             return f"Question: {question}\nAnswer: I don’t have enough information to answer this question."

#         max_tokens = 1024
#         tokens = self.tokenizer(context, return_tensors="pt")["input_ids"]
#         if tokens.shape[1] > max_tokens:
#             context = self.tokenizer.decode(tokens[0, -max_tokens:], skip_special_tokens=True)

#         raw_output = self.chain.invoke({"context": context, "question": question}).strip()
#         answer = raw_output.split("Answer:")[-1].strip()

#         return f"Question: {question}\nAnswer: {answer}"

import torch
import threading
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline # type: ignore
from langchain_core.prompts import PromptTemplate # type: ignore

class LLMReasoner:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens: int = 256):
        self._lock = threading.Lock()
        try:
            self.model_name = model_name
            self.max_new_tokens = max_new_tokens

            if torch.backends.mps.is_available():
                self.device = "mps"
                dtype = torch.float16
            elif torch.cuda.is_available():
                self.device = 0
                dtype = torch.float16
            else:
                self.device = -1
                dtype = torch.float32

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                low_cpu_mem_usage=True
            )

            hf_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

            self.prompt = PromptTemplate(
                input_variables=[
                    "system_prompt", "context", "query",
                    "intent", "emotion", "urgency", "complexity", "answer_size"
                ],
                template=(
                    "{system_prompt}\n\n"
                    "Context:\n{context}\n\n"
                    "Query: {query}\n\n"
                    "Features: {intent}, {emotion}, {urgency}, {complexity}\n\n"
                    "Guidance: {answer_size}\n\n"
                    "Answer:"
                )
            )
            self.chain = self.prompt | self.llm
        except Exception as e:
            raise RuntimeError(f"LLM Initialization Failed: {e}")

    def invoke(self, inputs: Dict[str, Any]) -> str:
        with self._lock:
            try:
                query = inputs.get("query", "").strip()
                context = inputs.get("context", "").strip()
                
                max_context_len = 512
                if len(context) > 2000:
                    tokens = self.tokenizer(context, return_tensors="pt", truncation=True, max_length=max_context_len)["input_ids"]
                    context = self.tokenizer.decode(tokens[0], skip_special_tokens=True)

                llm_input = {
                    "system_prompt": inputs.get("system_prompt", "You are a polite support assistant."),
                    "context": context,
                    "query": query,
                    "intent": inputs.get("intent", "unknown"),
                    "emotion": inputs.get("emotion", "neutral"),
                    "urgency": inputs.get("urgency", "low"),
                    "complexity": inputs.get("complexity", "small"),
                    "answer_size": inputs.get("answer_size", "short"),
                }

                raw_output = self.chain.invoke(llm_input)
                
                if isinstance(raw_output, str):
                    processed = raw_output.split("Answer:")[-1].strip()
                    return processed
                return str(raw_output)

            except Exception as e:
                return f"Error generating response: {str(e)}"