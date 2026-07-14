# 🤖 NovaSphere Customer Support RAG Agent

An AI-powered customer support assistant that uses **Retrieval-Augmented Generation (RAG)** to answer customer queries from a local knowledge base while maintaining conversational context across interactions.

Built with **LangChain**, **Google Gemini**, **FAISS**, and **Streamlit**.

---

## 🚀 Live Demo

**Application:** https://novasphere-customer-support-rag-agent.streamlit.app/

---

## Features

- AI-powered customer support assistant
- Retrieval-Augmented Generation (RAG)
- Local PDF knowledge base
- Semantic search using FAISS
- Google Gemini 2.5 Flash integration
- Conversational memory
- Handles follow-up questions naturally
- Rejects out-of-domain queries gracefully
- Streamlit web interface
- Cached models and vector store for faster responses

---

## Tech Stack

| Component | Technology |
|----------|------------|
| Frontend | Streamlit |
| LLM | Google Gemini 2.5 Flash |
| Framework | LangChain |
| Embeddings | BAAI/bge-small-en-v1.5 |
| Vector Store | FAISS |
| Knowledge Base | Local PDF |
| Language | Python |

---

## System Architecture

```text
                    User
                      │
                      ▼
             Streamlit Chat UI
                      │
                      ▼
         Conversational Memory
                      │
                      ▼
              Query Retriever
                      │
                      ▼
              FAISS Vector Store
                      │
                      ▼
         Relevant Knowledge Chunks
                      │
                      ▼
          Prompt + Retrieved Context
                      │
                      ▼
          Google Gemini 2.5 Flash
                      │
                      ▼
               Answer Citations
```

---

## Project Structure

```text
Rag_agent/
│
├── app.py
├── requirements.txt
├── README.md
│
├── backend/
│   ├── loader.py
│   ├── embeddings.py
│   ├── retriever.py
│   ├── rag_chain.py
│   └── memory.py
│
├── data/
│   └── NovaSphere_Knowledge_Base.pdf
│
└── vectorstore/
    ├── index.faiss
    └── index.pkl
```

---

## Installation

Clone the repository

```bash
https://github.com/Devesh5553/Customer-Support-RAG-Agent.git

cd Customer-Support-RAG-Agent
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
GOOGLE_API_KEY=YOUR_GEMINI_API_KEY
```

Generate embeddings (only required if the vector store is not already present)

```bash
python backend/embeddings.py
```

Run the application

```bash
python -m streamlit run app.py
```

---

## Example Conversation

### User

```
Do you offer Premium Care?
```

### Assistant

```
Yes. Premium Care includes:

• Priority customer support
• Smart Warranty
• 2TB secure cloud storage
• Expedited shipping
• Early product access
```


---

### Follow-up Question

**User**

```
What are its benefits?
```

The assistant correctly understands that **"its"** refers to **Premium Care** by using conversational memory.

---

## RAG Workflow

1. Load the PDF knowledge base.
2. Split the document into FAQ entries.
3. Generate embeddings using Hugging Face.
4. Store vectors in FAISS.
5. Retrieve the most relevant FAQ entries.
6. Send the retrieved context to Gemini.
7. Generate a grounded response with source citations.
8. Maintain conversation history for contextual follow-up questions.

---


## Future Improvements

- Multi-user chat sessions
- Persistent memory with a database
- Hybrid keyword + semantic search
- Streaming responses
- Admin dashboard for knowledge management
- Docker support
- Authentication
- Cloud vector databases (Pinecone/Chroma)

---

## License

This project was developed for learning and demonstration purposes.
