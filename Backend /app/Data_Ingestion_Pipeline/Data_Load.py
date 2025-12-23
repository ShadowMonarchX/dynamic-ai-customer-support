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

class DataSource:
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.data = []

    def load_data(self):
        self.data = []
        for filename in os.listdir(self.folder_path):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.folder_path, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    self.data.append(f.read())

    def get_data(self):
        return self.data


folder = "data"
source = DataSource(folder)
source.load_data()
texts = source.get_data()
for t in texts:
    print(t)
