from dotenv import load_dotenv

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from backend.retriever import retrieve_documents
from backend.memory import get_session_history
from langchain_core.runnables.history import RunnableWithMessageHistory
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are NovaSphere Technologies' customer support assistant.

Answer ONLY from the retrieved context.

If the answer isn't present, say:

"I couldn't find that information in the knowledge base."

Always answer professionally.
"""
        ),
        ("human", "Context:\n{context}"),
        ("placeholder", "{history}"),
        ("human", "{question}")
    ]
)

chain = prompt | llm

conversation = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)

def ask(question: str, session_id: str = "default"):

    docs = retrieve_documents(question)

    context = "\n\n".join(
        f"{doc.metadata['faq_id']}\n{doc.page_content}"
        for doc in docs
    )

    response = conversation.invoke(
        {
            "question": question,
            "context": context,
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    sources = [
        doc.metadata["faq_id"]
        for doc in docs
    ]

    return response.content, sources

if __name__ == "__main__":

    while True:

        q = input("\nAsk: ")

        if q.lower() == "exit":
            break

        answer, sources = ask(q)

        print("\nAnswer:\n")
        print(answer)

