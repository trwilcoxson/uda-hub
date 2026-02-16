"""LangGraph state schema for UDA-Hub agent orchestration."""
from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """State schema for the UDA-Hub supervisor graph.

    Extends MessagesState which provides the 'messages' key with
    add-message reducer semantics.
    """
    next: str = ""
    ticket_id: str = ""
    user_email: str = ""
