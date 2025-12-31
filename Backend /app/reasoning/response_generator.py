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
        """
        Generate grounded response based on:
        - User query
        - Context (retrieved docs)
        - System prompt strategy
        """
        with self._lock:
            try:
                query = data.get("query", "")
                context_text = data.get("context", "")
                
                # 1️⃣ Token size check
                token_count = len(self.reasoner.tokenizer.encode(query))
                if token_count > self.max_query_size:
                    return "Error: Your question is too long. Please simplify."

                # 2️⃣ Dynamic response size guidance
                size_guidance = self._determine_size_constraint(query, data.get("intent", ""))

                # 3️⃣ Strict grounding enforcement
                system_prompt = (
                    f"{data.get('system_prompt', '')} {size_guidance} "
                    "ONLY ANSWER BASED ON THE PROVIDED CONTEXT. "
                    "DO NOT GUESS. IF THE ANSWER IS NOT IN CONTEXT, "
                    "RESPOND WITH: 'I’m not fully sure. Could you please clarify?'"
                )

                # 4️⃣ Fact-coverage self-check (pass context)
                full_input = {
                    "query": query,
                    "context": context_text,
                    "system_prompt": system_prompt,
                    "intent": data.get("intent"),
                    "emotion": data.get("emotion"),
                    "urgency": data.get("urgency"),
                    "follow_up": data.get("follow_up", False)
                }

                # 5️⃣ Invoke LLM reasoner
                answer = self.reasoner.invoke(full_input)

                # 6️⃣ Optional: validate length vs size guidance
                if len(answer.split()) < 3 and len(query.split()) > 2:
                    return "I’m not fully sure. Could you please clarify?"

                return answer

            except Exception as e:
                return f"Response Generation Failed: {str(e)}"
