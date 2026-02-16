from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentic.config import LLM_MODEL
from agentic.tools.classification_tools import classify_ticket

TRIAGE_SYSTEM_PROMPT = """You are the Triage Agent for CultPass customer support.

Your job is to classify incoming support tickets by analyzing the customer's message.
Use the classify_ticket tool to determine:
- issue_type: account, subscription, reservation, billing, technical, or general
- priority: low, medium, high, or urgent
- sentiment: positive, neutral, negative, or frustrated
- requires_human: whether this needs human escalation

Always call the classify_ticket tool with the customer's message.
Return the classification results clearly so the supervisor can route appropriately.
"""


def create_triage_agent():
    return create_react_agent(
        model=ChatOpenAI(model=LLM_MODEL),
        tools=[classify_ticket],
        prompt=TRIAGE_SYSTEM_PROMPT,
        name="triage_agent",
    )
