# import re
# import string
# from typing import List
# from langchain_core.documents import Document # type: ignore
# from .data_load import DataSource

# STOPWORDS = {
#     'i','me','my','myself','we','our','ours','ourselves','you','your','yours',
#     'yourself','yourselves','he','him','his','himself','she','her','hers','herself',
#     'it','its','itself','they','them','their','theirs','themselves','what','which',
#     'who','whom','this','that','these','those','am','is','are','was','were','be',
#     'been','being','have','has','had','having','do','does','did','doing','a','an',
#     'the','and','but','if','or','because','as','until','while','of','at','by','for',
#     'with','about','against','between','into','through','during','before','after',
#     'above','below','to','from','up','down','in','out','on','off','over','under',
#     'again','further','then','once','here','there','when','where','why','how','all',
#     'any','both','each','few','more','most','other','some','such','no','nor','not',
#     'only','own','same','so','than','too','very','s','t','can','will','just','don',
#     'should','now'
# }

# _punct_regex = re.compile(f"[{re.escape(string.punctuation)}]")

# class Preprocessor:
#     def __init__(self, documents: List[Document] = None):
#         self.documents = documents or []

#     def transform_documents(self, documents: List[Document] = None) -> List[Document]:
#         docs_to_process = documents or self.documents
#         seen = set()
#         processed_docs = []

#         for doc in docs_to_process:
#             text = doc.page_content.lower()
#             text = _punct_regex.sub("", text)
#             tokens = [t for t in text.split() if t not in STOPWORDS]
#             cleaned = " ".join(tokens)

#             if cleaned and cleaned not in seen:
#                 seen.add(cleaned)
#                 processed_docs.append(
#                     Document(
#                         page_content=cleaned,
#                         metadata=doc.metadata
#                     )
#                 )

#         return processed_docs


import re
import threading
from typing import List, Dict, Any
from langchain_core.documents import Document # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # Use underscore and an 's' # type: ignore

class Preprocessor:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150):
        self._lock = threading.Lock()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n### ", "\n\n", "\n", " ", ""]
        )
        self.noise_pattern = re.compile(r'-{3,}')
        self.header_pattern = re.compile(r'(### .*)')

    def _clean_content(self, text: str) -> str:
        try:
            text = self.noise_pattern.sub('', text)
            text = text.replace("shipping cost", "Shipping Charges")
            text = text.replace("delivery fee", "Shipping Charges")
            text = text.replace("postage", "Shipping Charges")
            return re.sub(r'\s+', ' ', text).strip()
        except Exception as e:
            raise RuntimeError(f"Cleaning Error: {e}")

    def _derive_metadata(self, header: str) -> Dict[str, Any]:
        meta = {"category": "general", "urgency": "low"}
        header_upper = header.upper()
        if "REFUND" in header_upper:
            meta.update({"category": "billing", "urgency": "high", "topic": "refund"})
        elif "TECHNICAL" in header_upper or "SUPPORT" in header_upper:
            meta.update({"category": "technical", "urgency": "medium"})
        return meta

    def transform_documents(self, documents: List[Document]) -> List[Document]:
        with self._lock:
            processed_chunks = []
            try:
                for doc in documents:
                    raw_text = doc.page_content
                    sections = self.header_pattern.split(raw_text)
                    
                    current_meta = doc.metadata.copy()
                    
                    for section in sections:
                        section = section.strip()
                        if not section:
                            continue
                        
                        if section.startswith("###"):
                            current_meta.update(self._derive_metadata(section))
                            continue
                        
                        cleaned_text = self._clean_content(section)
                        if not cleaned_text:
                            continue

                        chunks = self.splitter.split_text(cleaned_text)
                        for chunk in chunks:
                            processed_chunks.append(
                                Document(
                                    page_content=chunk,
                                    metadata=current_meta.copy()
                                )
                            )
                return processed_chunks
            except Exception as e:
                raise RuntimeError(f"Transformation Pipeline Failed: {e}")

    def get_stats(self, docs: List[Document]) -> str:
        return f"Total Chunks Generated: {len(docs)}"