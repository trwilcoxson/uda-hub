from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from agentic.config import LLM_MODEL
from agentic.tools.knowledge_tools import search_knowledge, get_article_by_id

KNOWLEDGE_SYSTEM_PROMPT = """You are the Knowledge Agent for CultPass customer support.

Your job is to find relevant information from the CultPass knowledge base to answer
customer questions. Use the search_knowledge tool to find articles related to the
customer's query.

Guidelines:
- Always search the knowledge base before answering
- Cite the article title when providing information
- If confidence scores are low (no confident match), clearly state that you couldn't
  find a definitive answer and recommend escalation to human support
- Use the get_article_by_id tool if you need the full content of a specific article
- Provide the suggested phrasing from articles when available
- Be helpful and conversational, not robotic
"""


def create_knowledge_agent():
    return create_react_agent(
        model=ChatOpenAI(model=LLM_MODEL),
        tools=[search_knowledge, get_article_by_id],
        prompt=KNOWLEDGE_SYSTEM_PROMPT,
        name="knowledge_agent",
    )
