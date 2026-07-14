from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

# -----------------------------
# Embedding Model
# -----------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# -----------------------------
# Load FAISS Index
# -----------------------------
db = FAISS.load_local(
    str(VECTORSTORE_PATH),
    embeddings,
    allow_dangerous_deserialization=True
)

# -----------------------------
# Create Retriever
# -----------------------------
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)


def retrieve_documents(query: str):
    """
    Retrieve top-k relevant documents.
    """
    return retriever.invoke(query)


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

            # Metadata
            if doc.metadata:
                print("\nMetadata:")
                for key, value in doc.metadata.items():
                    print(f"{key}: {value}")

            print("\nContent:\n")
            print(doc.page_content)
            print("=" * 70)