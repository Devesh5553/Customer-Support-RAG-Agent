from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from backend.loader import load_and_split
# -----------------------------
# Paths
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

PDF_PATH = BASE_DIR / "data" / "NovaSphere_Knowledge_Base.pdf"
VECTORSTORE_PATH = BASE_DIR / "vectorstore"

# -----------------------------
# Load and Split Documents
# -----------------------------
print("Loading knowledge base...")
documents = load_and_split(str(PDF_PATH))

print(f"Loaded {len(documents)} chunks.")

# -----------------------------
# Embedding Model
# -----------------------------
print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# -----------------------------
# Build FAISS Index
# -----------------------------
print("Creating FAISS index...")

db = FAISS.from_documents(documents, embeddings)

# -----------------------------
# Save
# -----------------------------
db.save_local(str(VECTORSTORE_PATH))

print("\nVector store created successfully!")
print(f"Saved at: {VECTORSTORE_PATH}")