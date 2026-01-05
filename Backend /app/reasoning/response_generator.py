import threading
from typing import Dict, Any
from app.reasoning.llm_reasoner import LLMReasoner


class ResponseGenerator:
    def __init__(self, model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
        self._lock = threading.Lock()
        self.reasoner = LLMReasoner(model_name=model_name)
        self.max_query_tokens = 500

    def _determine_size_constraint(self, query: str, intent: str) -> str:
        words = len(query.split())
        if intent == "greeting" or words <= 2:
            return "Reply in 3 to 5 words."
        if words < 10:
            return "Reply in one concise sentence."
        return "Reply in 2 to 3 professional sentences."

    def generate(self, data: Dict[str, Any]) -> str:
        with self._lock:
            query = str(data.get("query", "")).strip()
            context = str(data.get("context", "")).strip()
            if (
                not query
                or not context
                or not getattr(self.reasoner, "tokenizer", None)
            ):
                return "I’m not fully sure. Could you please clarify?"

            try:
                token_count = len(self.reasoner.tokenizer.encode(query))
            except Exception:
                token_count = len(query.split())
            if token_count > self.max_query_tokens:
                return "Your question is too long. Please simplify it."

            size_rule = self._determine_size_constraint(
                query, str(data.get("intent", "unknown"))
            )
            system_prompt = (
                f"{str(data.get('system_prompt', ''))} {size_rule} "
                "Answer strictly using the provided context. If the answer is missing, say you are not sure."
            )

            llm_input = {
                "query": query,
                "context": context,
                "system_prompt": system_prompt,
                "intent": str(data.get("intent", "unknown")),
                "emotion": str(data.get("emotion", "neutral")),
                "urgency": str(data.get("urgency", "low")),
                "complexity": str(data.get("complexity", "small")),
                "answer_size": size_rule,
            }

            try:
                answer = self.reasoner.invoke(llm_input)
            except Exception:
                return "I’m not fully sure. Could you please clarify?"

            if not answer or len(answer.split()) < 3:
                return "I’m not fully sure. Could you please clarify()"

            return str(answer).strip()
