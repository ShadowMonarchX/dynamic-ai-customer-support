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