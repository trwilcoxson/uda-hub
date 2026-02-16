import logging

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph_supervisor import create_supervisor

from agentic.config import LLM_MODEL
from agentic.logging_config import log_structured
from agentic.agents.triage import create_triage_agent
from agentic.agents.knowledge import create_knowledge_agent
from agentic.agents.account import create_account_agent
from agentic.agents.action import create_action_agent

logger = logging.getLogger(__name__)

SUPERVISOR_PROMPT = (
    "You are the Supervisor for CultPass customer support. "
    "You orchestrate specialized agents to resolve customer tickets.\n\n"
    "Available agents:\n"
    "- triage_agent: Classifies tickets by issue type, priority, sentiment, and urgency. "
    "Always route new tickets here first.\n"
    "- knowledge_agent: Searches the knowledge base (RAG) for answers to general "
    "questions about CultPass (subscriptions, reservations, policies, troubleshooting).\n"
    "- account_agent: Looks up customer account information (user profile, subscription "
    "details, reservation history). Use when the customer asks about their specific account.\n"
    "- action_agent: Performs write operations (cancel reservations, process refunds, "
    "update subscriptions). Use when the customer wants to make changes.\n\n"
    "Routing rules:\n"
    "1. For every new customer message, first route to triage_agent for classification.\n"
    "2. Use the triage classification result to make routing decisions:\n"
    "   - issue_type determines which agent to route to\n"
    "   - sentiment determines tone and urgency of handling:\n"
    "     * 'frustrated' or 'negative' sentiment: prioritize speed, be extra empathetic, "
    "       and if combined with high/urgent priority, consider escalation\n"
    "     * 'positive' sentiment: acknowledge their positivity in your response\n"
    "   - priority determines order: 'urgent' and 'high' tickets should be handled immediately\n"
    "3. Route by issue_type after triage:\n"
    "   - 'general' or 'technical' -> knowledge_agent\n"
    "   - 'account' -> account_agent (then knowledge_agent if needed)\n"
    "   - 'subscription', 'reservation' -> account_agent first to check status, "
    "     then action_agent if changes requested\n"
    "   - 'billing' -> knowledge_agent for policy, then action_agent for refunds\n"
    "4. If triage indicates requires_human=true, compose an empathetic escalation response.\n"
    "5. If sentiment is 'frustrated', always acknowledge their frustration before resolving.\n"
    "6. After getting results from agents, compose a helpful final response to the customer.\n\n"
    "Always be friendly, professional, and concise in your responses."
)


def build_graph():
    """Build and compile the supervisor graph with all worker agents."""
    triage = create_triage_agent()
    knowledge = create_knowledge_agent()
    account = create_account_agent()
    action = create_action_agent()

    checkpointer = MemorySaver()

    supervisor = create_supervisor(
        model=ChatOpenAI(model=LLM_MODEL),
        agents=[triage, knowledge, account, action],
        prompt=SUPERVISOR_PROMPT,
    )

    app = supervisor.compile(checkpointer=checkpointer)
    log_structured(logger, "UDA-Hub supervisor graph compiled successfully",
                   agent="supervisor", action="build_graph")
    return app


# Build the app at module level for import
app = build_graph()
