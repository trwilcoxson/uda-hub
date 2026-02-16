from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentic.config import LLM_MODEL
from agentic.tools.account_tools import lookup_user, get_subscription, get_reservations
from agentic.tools.memory_tools import get_customer_context, record_customer_preference

ACCOUNT_SYSTEM_PROMPT = """You are the Account Agent for CultPass customer support.

Your job is to look up customer account information from the CultPass database
and provide personalized support using the customer's history.

Guidelines:
- Use lookup_user to find a customer by their email address
- Use get_subscription to check their subscription status and tier
- Use get_reservations to view their booking history
- After identifying a customer, use get_customer_context with their user_id to load
  their past interactions, resolved issues, and preferences from long-term memory.
  Use this context to personalize your response (e.g., acknowledge returning customers,
  reference past issues, respect their preferences).
- Use record_customer_preference to save any preferences the customer mentions
  (e.g., preferred language, contact method)
- Always verify the customer's identity by email before sharing account details
- If a user's account is blocked, note this clearly in your response
- Never modify any data - you are read-only (except for storing preferences)
- Present information clearly and concisely
"""


def create_account_agent():
    return create_react_agent(
        model=ChatOpenAI(model=LLM_MODEL),
        tools=[
            lookup_user, get_subscription, get_reservations,
            get_customer_context, record_customer_preference,
        ],
        prompt=ACCOUNT_SYSTEM_PROMPT,
        name="account_agent",
    )
