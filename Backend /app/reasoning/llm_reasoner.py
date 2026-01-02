# import torch # type: ignore
# import threading
# from typing import Dict, Any
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline # type: ignore
# from langchain_huggingface import HuggingFacePipeline # type: ignore
# from langchain_core.prompts import PromptTemplate # type: ignore

# class LLMReasoner:
#     def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0", max_new_tokens: int = 256):
#         self._lock = threading.Lock()
#         try:
#             self.model_name = model_name
#             self.max_new_tokens = max_new_tokens

#             # Device selection logic
#             if torch.backends.mps.is_available():
#                 self.device = "mps"
#                 dtype = torch.float16
#             elif torch.cuda.is_available():
#                 self.device = 0
#                 dtype = torch.float16
#             else:
#                 self.device = -1
#                 dtype = torch.float32

#             self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
#             # Fixed: Changed torch_dtype to dtype to remove deprecation warning
#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 dtype=dtype, 
#                 low_cpu_mem_usage=True
#             )

#             hf_pipeline = pipeline(
#                 "text-generation",
#                 model=self.model,
#                 tokenizer=self.tokenizer,
#                 device=self.device,
#                 max_new_tokens=self.max_new_tokens,
#                 do_sample=True,
#                 temperature=0.7,
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
#             self.llm = HuggingFacePipeline(pipeline=hf_pipeline)

#             # Updated Prompt Template to emphasize the 'Guidance' (Answer Size)
#             self.prompt = PromptTemplate(
#                 input_variables=[
#                     "system_prompt", "context", "query",
#                     "intent", "emotion", "urgency", "complexity", "answer_size"
#                 ],
#                 template=(
#                     "<|system|>\n"
#                     "{system_prompt}\n"
#                     "Constraint: {answer_size}\n"
#                     "Context: {context}</s>\n"
#                     "<|user|>\n"
#                     "Query: {query}\n"
#                     "Status: Intent={intent}, Emotion={emotion}, Urgency={urgency}</s>\n"
#                     "<|assistant|>\n"
#                     "Answer:"
#                 )
#             )
#             self.chain = self.prompt | self.llm
#         except Exception as e:
#             raise RuntimeError(f"LLM Initialization Failed: {e}")

#     def invoke(self, inputs: Dict[str, Any]) -> str:
#         # Thread safety lock
#         with self._lock:
#             try:
#                 query = inputs.get("query", "").strip()
#                 context = inputs.get("context", "").strip()
                
#                 # Dynamic context truncation based on tokenizer
#                 max_context_tokens = 512
#                 context_ids = self.tokenizer.encode(context, truncation=True, max_length=max_context_tokens)
#                 context = self.tokenizer.decode(context_ids, skip_special_tokens=True)

#                 # Preparation of inputs for the chain
#                 llm_input = {
#                     "system_prompt": inputs.get("system_prompt", "You are a professional assistant."),
#                     "context": context,
#                     "query": query,
#                     "intent": inputs.get("intent", "unknown"),
#                     "emotion": inputs.get("emotion", "neutral"),
#                     "urgency": inputs.get("urgency", "low"),
#                     "complexity": inputs.get("complexity", "small"),
#                     "answer_size": inputs.get("answer_size", "Provide a concise response.")
#                 }

#                 # Execution
#                 raw_output = self.chain.invoke(llm_input)
                
#                 # Robust cleaning of output
#                 if isinstance(raw_output, str):
#                     # Splitting to ensure we only return the AI's actual answer
#                     processed = raw_output.split("Answer:")[-1].strip()
#                     # Remove any leftover stop tokens or stray system artifacts
#                     processed = processed.replace("</s>", "").replace("<|assistant|>", "").strip()
#                     return processed
                
#                 return str(raw_output)

#             except Exception as e:
#                 return f"Error in reasoning chain: {str(e)}"



import torch
import threading
from typing import Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

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
                model_name, torch_dtype=dtype, low_cpu_mem_usage=True
            )

            hf_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device,
                max_new_tokens=self.max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            self.llm = HuggingFacePipeline(pipeline=hf_pipeline)
            self.prompt = PromptTemplate(
                input_variables=["system_prompt","context","query","intent","emotion","urgency","complexity","answer_size"],
                template=(
                    "<|system|>\n{system_prompt}\nConstraint: {answer_size}\nContext:\n{context}\n</s>\n"
                    "<|user|>\n{query}\nIntent={intent}, Emotion={emotion}, Urgency={urgency}\n</s>\n"
                    "<|assistant|>\nAnswer:"
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
                if not query or not context:
                    return "I’m not fully sure. Could you please clarify?"

                context_ids = self.tokenizer.encode(context, truncation=True, max_length=512)
                context = self.tokenizer.decode(context_ids, skip_special_tokens=True)

                llm_input = {
                    "system_prompt": inputs.get("system_prompt","You are a professional assistant."),
                    "context": context,
                    "query": query,
                    "intent": inputs.get("intent", "unknown"),
                    "emotion": inputs.get("emotion", "neutral"),
                    "urgency": inputs.get("urgency", "low"),
                    "complexity": inputs.get("complexity", "small"),
                    "answer_size": inputs.get("answer_size","Provide a concise response.")
                }

                output = self.chain.invoke(llm_input)
                if isinstance(output, str):
                    text = output.split("Answer:")[-1].replace("</s>", "").replace("<|assistant|>", "").strip()
                    return text

                return str(output)
            except Exception:
                return "I’m not fully sure. Could you please clarify?"
