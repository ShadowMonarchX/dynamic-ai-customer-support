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

