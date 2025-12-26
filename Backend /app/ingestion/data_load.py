# from pathlib import Path
# import pandas as pd
# import json
# from datetime import datetime

# class LongLivedKnowledge:
#     def __init__(self, paths):
#         self.paths = paths
#         self.store = "Long-Lived Data Loader"
#         self.data = []

#     def load(self):
#         for path in self.paths:
#             folder = Path(path)
#             for file in folder.glob("*.*"):
#                 with open(file, "r", encoding="utf-8") as f:
#                     content = f.read()
#                     self.data.append({
#                         "source": str(file),
#                         "content": content,
#                         "timestamp": datetime.now()
#                     })
#         return self.data

# class OperationalKnowledge:
#     def __init__(self, paths):
#         self.paths = paths
#         self.store = "Mid-Lived Data Loader"
#         self.data = []

#     def load(self):
#         for path in self.paths:
#             if path.endswith(".csv"):
#                 df = pd.read_csv(path)
#                 self.data.extend(df.to_dict(orient="records"))
#             elif path.endswith(".json"):
#                 with open(path, "r", encoding="utf-8") as f:
#                     self.data.extend(json.load(f))
#         return self.data

# class LiveContextualData:
#     def __init__(self, paths):
#         self.paths = paths
#         self.store = "Ephemeral Context Data Loader"
#         self.data = []

#     def load(self):
#         for path in self.paths:
#             with open(path, "r", encoding="utf-8") as f:
#                 self.data.extend(json.load(f))
#         return self.data

# long_lived_paths = ["data/policies/", "data/manuals/", "data/legal_docs/", "data/guides/"]
# operational_paths = ["data/product_info.csv", "data/pricing.csv", "data/faqs.json", "data/business_rules.csv"]
# live_data_paths = ["data/live_stock.json", "data/live_orders.json", "data/sessions.json", "data/user_context.json"]

# long_lived_loader = LongLivedKnowledge(long_lived_paths)
# operational_loader = OperationalKnowledge(operational_paths)
# live_loader = LiveContextualData(live_data_paths)

# long_lived_data = long_lived_loader.load()
# operational_data = operational_loader.load()
# live_data = live_loader.load()

# print(f"Loaded {len(long_lived_data)} long-lived documents")
# print(f"Loaded {len(operational_data)} operational records")
# print(f"Loaded {len(live_data)} live/contextual records")

import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader # type: ignore
from langchain_core.documents import Document # type: ignore


class DataSource:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self.documents: list[Document] = []

    def load_data(self) -> None:
        try:
            if os.path.isfile(self.path) and self.path.endswith(".txt"):
                loader = TextLoader(self.path, encoding="utf-8")
                self.documents = loader.load()

            elif os.path.isdir(self.path):
                loader = DirectoryLoader(
                    self.path,
                    glob="**/*.txt",
                    loader_cls=TextLoader,
                    loader_kwargs={"encoding": "utf-8"},
                )
                self.documents = loader.load()

            else:
                raise ValueError(
                    f"Invalid path or unsupported format: {self.path}"
                )

        except Exception as e:
            raise RuntimeError(f"Failed to load data: {e}")

    def get_documents(self) -> list[Document]:
        return self.documents

