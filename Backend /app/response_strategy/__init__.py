from .greeting_response import GreetingResponse
from .emotion_response import EmotionResponse
from .faq_response import FAQResponse
from .transactional_response import TransactionalResponse
from .big_issue_response import BigIssueResponse

def select_response_strategy(features: dict) -> str:
    intent = features.get("intent", "unknown")
    emotion = features.get("emotion", "neutral")
    
    # Priority 1: High Emotion
    if emotion in ["angry", "frustrated"]:
        return EmotionResponse().get_strategy(features)
    
    # Priority 2: Intent-based
    strategies = {
        "greeting": GreetingResponse(),
        "faq": FAQResponse(),
        "transactional": TransactionalResponse(),
        "big_issue": BigIssueResponse(),
        "account_support": BigIssueResponse()
    }
    
    selected = strategies.get(intent, FAQResponse())
    return selected.get_strategy(features)