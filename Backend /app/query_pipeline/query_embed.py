# ## Step 3: Intelligent Retrieval from Your Data

# ### Folder

# `query_pipeline/`

# ### Files Involved

# * `query_embed.py`
# * `retrieval_router.py`

# ### What Happens Here

# The bot searches **only your internal data**, such as:

# * Website FAQs
# * Help center articles
# * Order and delivery policies
# * Past support tickets
# * Product documentation
# * App database summaries

# The system:

# * Retrieves **multiple relevant documents**
# * Ranks them by **semantic relevance**
# * Filters **outdated or irrelevant** information


# 📌 This step is critical for **hallucination prevention**.
class QueryEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self._lock = threading.Lock()
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    def embed_documents(self, documents):
        with self._lock:
            # convert to Document objects if needed
            docs = [
                (
                    d
                    if isinstance(d, Document)
                    else Document(page_content=str(d), metadata={})
                )
                for d in documents
            ]
            texts, metas = [], []
            for doc in docs:
                meta = dict(doc.metadata or {})
                texts.append(self._apply_bias(doc.page_content, meta))
                meta["confidence_weight"] = self._confidence_weight(meta)
                doc.metadata = meta
                metas.append(meta)
            embeddings = np.atleast_2d(
                np.array(self.embedding_model.embed_documents(texts), dtype=np.float32)
            )
            faiss.normalize_L2(embeddings)
            return embeddings
