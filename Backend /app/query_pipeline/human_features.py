from langdetect import detect, DetectorFactory # type: ignore

DetectorFactory.seed = 0

URGENT_KEYWORDS = ["now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"]
FRUSTRATION_KEYWORDS = ["angry", "frustrated", "annoyed", "ridiculous", "worst", "not working", "failed"]

def extract_human_features(query: str) -> dict:
    lowered = query.lower()
    urgency_flag = any(word in lowered for word in URGENT_KEYWORDS)
    is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)
    try:
        language = detect(query)
    except:
        language = "unknown"
    return {
        "urgency": "high" if urgency_flag else "low",
        "emotion": "frustrated" if is_frustrated else "neutral",
        "sentiment_score": -0.8 if is_frustrated else 0.0,
        "language": language
    }
