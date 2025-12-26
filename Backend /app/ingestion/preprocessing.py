import re
import string
from typing import List
from langchain_core.documents import Document # type: ignore


STOPWORDS = {
    'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
    'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
    'it','its','itself','they','them','their','theirs','themselves','what','which',
    'who','whom','this','that','these','those','am','is','are','was','were','be',
    'been','being','have','has','had','having','do','does','did','doing','a','an',
    'the','and','but','if','or','because','as','until','while','of','at','by','for',
    'with','about','against','between','into','through','during','before','after',
    'above','below','to','from','up','down','in','out','on','off','over','under',
    'again','further','then','once','here','there','when','where','why','how','all',
    'any','both','each','few','more','most','other','some','such','no','nor','not',
    'only','own','same','so','than','too','very','s','t','can','will','just','don',
    'should','now'
}

_punct_regex = re.compile(f"[{re.escape(string.punctuation)}]")


class Preprocessor:
    """
    LangChain-style document preprocessor:
    - Lowercase
    - Remove punctuation
    - Remove stopwords
    - Deduplicate content
    """

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        seen = set()
        processed_docs = []

        for doc in documents:
            text = doc.page_content.lower()
            text = _punct_regex.sub("", text)
            tokens = [t for t in text.split() if t not in STOPWORDS]
            cleaned = " ".join(tokens)

            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                processed_docs.append(
                    Document(
                        page_content=cleaned,
                        metadata=doc.metadata
                    )
                )

        return processed_docs
