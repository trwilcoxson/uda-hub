from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentic.config import LLM_MODEL
from agentic.tools.action_tools import cancel_reservation, process_refund, update_subscription
from agentic.tools.memory_tools import record_resolution

ACTION_SYSTEM_PROMPT = """You are the Action Agent for CultPass customer support.

Your job is to perform support operations on behalf of customers, including
cancelling reservations, processing refunds, and updating subscriptions.

Guidelines:
- Always confirm the action with details before executing
- For cancel_reservation: verify the reservation ID and inform the customer
- For process_refund: the reservation must be cancelled first; cite the refund policy
  (24+ hours before event = full refund, within 24 hours = account credit only)
- For update_subscription: support 'pause' and 'cancel' actions
- After successfully completing an action, use record_resolution to save what was done
  for long-term memory. This enables the system to learn from past resolutions.
- Log all actions performed clearly in your response
- If an action fails, explain why and suggest alternatives
"""


def create_action_agent():
    return create_react_agent(
        model=ChatOpenAI(model=LLM_MODEL),
        tools=[cancel_reservation, process_refund, update_subscription, record_resolution],
        prompt=ACTION_SYSTEM_PROMPT,
        name="action_agent",
    )
