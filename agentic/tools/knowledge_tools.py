"""Knowledge retrieval tools for the Knowledge Agent (RAG search)."""
import json
import logging

from langchain_core.tools import tool

from agentic.db import udahub_session
from agentic.logging_config import log_structured
from agentic.rag.retriever import retriever

logger = logging.getLogger(__name__)


@tool
def search_knowledge(query: str) -> str:
    """Search the CultPass knowledge base for articles relevant to a customer query.

    Uses RAG to find the most relevant support articles and returns them
    with confidence scores. Articles with confidence below the threshold
    are flagged for potential escalation.

    Args:
        query: The search query describing what the customer needs help with.
    """
    results, has_confident = retriever.search_above_threshold(query)

    if not results:
        return json.dumps({
            "articles": [],
            "has_confident_match": False,
            "message": "No matching articles found. Consider escalating to human support.",
        })

    articles_out = [
        {
            "article_id": r["article_id"],
            "title": r["title"],
            "content": r["content"],
            "confidence": r["confidence"],
        }
        for r in results
    ]

    response = {
        "articles": articles_out,
        "has_confident_match": has_confident,
    }
    if not has_confident:
        response["message"] = (
            "No high-confidence matches found. The results below may not fully "
            "address the customer's question. Consider escalating to human support."
        )

    log_structured(logger, "Knowledge base search completed",
                   agent="knowledge_agent", action="search_knowledge",
                   details={"query": query[:100], "results_count": len(articles_out),
                            "has_confident_match": has_confident,
                            "top_confidence": articles_out[0]["confidence"] if articles_out else None})
    return json.dumps(response)


@tool
def get_article_by_id(article_id: str) -> str:
    """Retrieve a specific knowledge base article by its ID.

    Args:
        article_id: The unique identifier of the article to retrieve.
    """
    from data.models.udahub import Knowledge

    with udahub_session() as session:
        article = session.query(Knowledge).filter_by(article_id=article_id).first()
        if not article:
            return json.dumps({"error": f"Article {article_id} not found"})

        return json.dumps({
            "article_id": article.article_id,
            "title": article.title,
            "content": article.content,
            "tags": article.tags,
        })
