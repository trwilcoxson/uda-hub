import logging
from typing import Optional
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from agentic.logging_config import log_structured
from agentic.config import (
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAG_CONFIDENCE_THRESHOLD,
    RAG_TOP_K,
)

logger = logging.getLogger(__name__)


class KnowledgeRetriever:
    """Retrieves relevant knowledge articles from ChromaDB with confidence scoring."""

    def __init__(self):
        self._client = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            embedding_fn = OpenAIEmbeddingFunction(
                model_name=EMBEDDING_MODEL,
                api_key=None,
            )
            self._client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
            self._collection = self._client.get_collection(
                name=CHROMA_COLLECTION_NAME,
                embedding_function=embedding_fn,
            )
        return self._collection

    def search(self, query: str, top_k: Optional[int] = None) -> list[dict]:
        """Search for relevant articles.

        Returns list of dicts with keys: article_id, title, content, tags, confidence
        Confidence is 1.0 - normalized_distance (higher = more relevant).
        """
        top_k = top_k or RAG_TOP_K
        collection = self._get_collection()

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        articles = []
        if not results["ids"][0]:
            return articles

        for i, doc_id in enumerate(results["ids"][0]):
            distance = results["distances"][0][i]
            # ChromaDB default uses L2 distance; convert to confidence score
            # Normalize: confidence = 1 / (1 + distance)
            confidence = 1.0 / (1.0 + distance)
            metadata = results["metadatas"][0][i]

            articles.append({
                "article_id": doc_id,
                "title": metadata.get("title", ""),
                "content": results["documents"][0][i],
                "tags": metadata.get("tags", ""),
                "confidence": round(confidence, 4),
            })

        log_structured(logger, "RAG similarity search completed",
                       agent="knowledge_agent", action="rag_search",
                       details={"query": query[:100], "results_count": len(articles),
                                "top_confidence": articles[0]["confidence"] if articles else None})
        return articles

    def search_above_threshold(
        self, query: str, top_k: Optional[int] = None
    ) -> tuple[list[dict], bool]:
        """Search and filter by confidence threshold.

        Returns (results, has_confident_match).
        """
        results = self.search(query, top_k)
        confident_results = [
            r for r in results if r["confidence"] >= RAG_CONFIDENCE_THRESHOLD
        ]
        return results, len(confident_results) > 0


# Module-level singleton
retriever = KnowledgeRetriever()
