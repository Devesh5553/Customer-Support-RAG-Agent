"""
loader.py

Loads the NovaSphere Knowledge Base and splits it into
one Document per FAQ entry.
"""

from pathlib import Path
import re

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(file_path: str):
    """
    Load a PDF or TXT file.
    """

    path = Path(file_path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Knowledge base not found: {path}")

    if path.suffix.lower() == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        raise ValueError("Only PDF are supported.")

    return loader.load()


def split_by_faq(documents):
    """
    Combine all pages and split into one document per FAQ.
    """

    full_text = "\n".join(doc.page_content for doc in documents)

    # Split whenever a new FAQ starts
    faq_sections = re.split(r"(?=FAQ-\d{3})", full_text)

    chunks = []

    for faq in faq_sections:

        faq = faq.strip()

        if not faq:
            continue

        # Extract FAQ number
        match = re.search(r"(FAQ-\d{3})", faq)
        faq_id = match.group(1) if match else "Unknown"

        chunks.append(
            Document(
                page_content=faq,
                metadata={
                    "faq_id": faq_id,
                    "source": "NovaSphere_Knowledge_Base.pdf",
                },
            )
        )

    return chunks


def load_and_split(file_path: str):

    documents = load_documents(file_path)

    return split_by_faq(documents)


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    FILE_PATH = BASE_DIR / "data" / "NovaSphere_Knowledge_Base.pdf"

    chunks = load_and_split(str(FILE_PATH))

    print(f"\nLoaded {len(chunks)} FAQ entries.\n")

    print("=" * 80)
    print(chunks[0].metadata)
    print("=" * 80)
    print(chunks[0].page_content)