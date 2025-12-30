import threading
from typing import Dict, Any
from app.reasoning.llm_reasoner import LLMReasoner

class ResponseGenerator:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self._lock = threading.Lock()
        self.reasoner = LLMReasoner(model_name=model_name)
        self.max_query_size = 500  # Token limit for safety

    def _determine_size_constraint(self, query: str, intent: str) -> str:
        """Sets strict word counts based on user input length and intent."""
        word_count = len(query.split())
        
        if word_count <= 2 or intent == "greeting":
            return "REPLY ONLY WITH 3 TO 5 WORDS. BE EXTREMELY BRIEF."
        elif word_count < 10:
            return "Keep your response to a single concise sentence."
        else:
            return "Provide a detailed but professional 2-3 sentence response."

    def generate(self, data: Dict[str, Any]) -> str:
        with self._lock:
            try:
                query = data.get("query", "")
                
                # 1. Size Check for Safety
                token_count = len(self.reasoner.tokenizer.encode(query))
                if token_count > self.max_query_size:
                    return "Error: Your question is too long. Please simplify."

                # 2. Dynamic Size Guidance
                size_guidance = self._determine_size_constraint(query, data.get("intent", ""))
                
                # 3. Assemble Final Prompt
                full_input = {
                    **data,
                    "answer_size": size_guidance,
                    "system_prompt": f"{data.get('system_prompt', '')} {size_guidance}"
                }

                # 4. Final Invoke
                return self.reasoner.invoke(full_input)

            except Exception as e:
                return f"Response Generation Failed: {str(e)}"