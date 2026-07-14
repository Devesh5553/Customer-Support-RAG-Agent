from langchain_core.chat_history import InMemoryChatMessageHistory

# Stores chat history for each session
store = {}


def get_session_history(session_id: str):
    """
    Returns the chat history for a session.
    Creates one if it doesn't exist.
    """

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]