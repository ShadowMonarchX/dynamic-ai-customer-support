from langdetect import detect, DetectorFactory  # type: ignore

DetectorFactory.seed = 0

URGENT_KEYWORDS = {"now", "urgent", "asap", "immediately", "today", "tomorrow", "right away"}
FRUSTRATION_KEYWORDS = {"angry", "frustrated", "annoyed", "ridiculous", "worst", "not working", "failed"}

def extract_human_features(query: str) -> dict:
    """
    Analyze a user query to extract human-centric features:
    - urgency
    - emotional state
    - sentiment score
    - language
    """
    lowered = query.lower()
    is_urgent = any(word in lowered for word in URGENT_KEYWORDS)
    is_frustrated = any(word in lowered for word in FRUSTRATION_KEYWORDS)
    
    try:
        language = detect(query)
    except:
        language = "unknown"
    
    return {
        "urgency": "high" if is_urgent else "low",
        "emotion": "frustrated" if is_frustrated else "neutral",
        "sentiment_score": -0.8 if is_frustrated else 0.0,
        "language": language
    }
