import uuid

import streamlit as st

from backend.rag_chain import ask

st.set_page_config(
    page_title="NovaSphere Support Assistant",
    page_icon="🤖",
    layout="centered",
)

# -----------------------------
# Session State
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# -----------------------------
# Sidebar
# -----------------------------

with st.sidebar:

    st.title("🤖 NovaSphere")

    st.markdown(
        """
### Customer Support RAG

This assistant answers questions using
NovaSphere's internal knowledge base.

**Features**

- RAG using FAISS
- Gemini 2.5 Flash
- Conversational Memory
- Source Citations
"""
    )

    st.divider()

    if st.button("🗑 Clear Conversation", use_container_width=True):

        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())

        st.rerun()

# -----------------------------
# Header
# -----------------------------

st.title("🤖 NovaSphere Customer Support")

st.caption(
    "Ask questions about memberships, shipping, returns, warranties, payments, and more."
)

# -----------------------------
# Welcome Message
# -----------------------------

if len(st.session_state.messages) == 0:

    st.info(
        """
Welcome!

I can answer questions about:

- Shipping
- Returns
- Memberships
- Warranty
- Payments
- Business Hours

Every answer is generated from the NovaSphere Knowledge Base.
"""
    )

# -----------------------------
# Display Chat History
# -----------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"],
        avatar="👤" if message["role"] == "user" else "🤖",
    ):

        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:
                    st.write(source)

# -----------------------------
# Chat Input
# -----------------------------

if prompt := st.chat_input("Ask your question..."):

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):

        with st.spinner("Searching knowledge base..."):

            answer, sources = ask(
                prompt,
                st.session_state.session_id,
            )

        # Remove duplicate sources while preserving order
        sources = list(dict.fromkeys(sources))

        st.markdown(answer)


    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )