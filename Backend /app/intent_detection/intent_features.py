
# ## Step 1: User Query Understanding

# ### Folder

# `intent_detection/`

# ### Files Involved

# * `intent_classifier.py`
# * `intent_features.py`

# ### What Happens Here

# The bot analyzes the raw user message to understand:

# * **User intent** (billing, delivery, refund, account issue)
# * **Question type** (status, how-to, policy, troubleshooting)
# * **Language clarity** (short, vague, emotional, follow-up)

# ### Example

# User query:

# > “My order is delayed, why?”

# Bot understands:

# * Topic: **Order / Delivery**
# * Intent: **Issue / Complaint**
# * Required data: **Order policy + delivery timelines**

# This step produces **intent and sentiment features**, not an answer.

