import bs4
from langchain_community.document_loaders import WebBaseLoader # type: ignore
from langchain_community.document_loaders.web_base import bs4  # type: ignore

# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

assert len(docs) == 1
print(f"Total characters: {len(docs[0].page_content)}")
print("\n--- Sample Content ---\n")
print(docs[0].page_content[:500])
print("\n... ...\n")


# for i, chunk in enumerate(docs[0].page_content.split("\n\n")[:5]):
#     print(f"\n--- Chunk {i+1} ---\n")
#     print(chunk)    

