from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"


# -----------------------------
# Cached Resources
# -----------------------------
@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5"
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    return FAISS.load_local(
        str(VECTORSTORE_PATH),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


@lru_cache(maxsize=1)
def get_retriever():
    return get_vectorstore().as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 10,
        },
    )


# -----------------------------
# Retrieval
# -----------------------------
def retrieve_documents(query: str):
    return get_retriever().invoke(query)


# -----------------------------
# Testing
# -----------------------------
if __name__ == "__main__":

    while True:

        question = input("\nAsk a question (type 'exit' to quit): ")

        if question.lower() == "exit":
            break

        docs = retrieve_documents(question)

        print("\nTop Retrieved Documents\n")

        for i, doc in enumerate(docs, start=1):

            print("=" * 70)
            print(f"Result {i}")

            if doc.metadata:
                print("\nMetadata:")
                for key, value in doc.metadata.items():
                    print(f"{key}: {value}")

            print("\nContent:\n")
            print(doc.page_content)

            print("=" * 70)