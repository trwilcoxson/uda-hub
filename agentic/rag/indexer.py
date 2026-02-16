import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from agentic.logging_config import log_structured
from agentic.config import (
    UDAHUB_DB_PATH,
    CHROMA_PERSIST_DIR,
    CHROMA_COLLECTION_NAME,
    EMBEDDING_MODEL,
)

logger = logging.getLogger(__name__)


def index_knowledge_base():
    """Load all Knowledge articles from udahub.db and embed them into ChromaDB."""
    from data.models.udahub import Knowledge

    engine = create_engine(f"sqlite:///{UDAHUB_DB_PATH}", echo=False)
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        articles = session.query(Knowledge).all()
        if not articles:
            log_structured(logger, "No knowledge articles found in udahub.db",
                       agent="indexer", action="index_knowledge_base", level=logging.WARNING)
            return 0

        embedding_fn = OpenAIEmbeddingFunction(
            model_name=EMBEDDING_MODEL,
            api_key=None,  # uses OPENAI_API_KEY env var
        )

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        # Delete existing collection if it exists, then recreate
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass  # Collection may not exist yet on first run

        collection = client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=embedding_fn,
        )

        ids = []
        documents = []
        metadatas = []

        for article in articles:
            ids.append(article.article_id)
            documents.append(f"{article.title}\n\n{article.content}")
            metadatas.append({
                "title": article.title,
                "tags": article.tags or "",
                "article_id": article.article_id,
            })

        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        log_structured(logger, f"Indexed {len(articles)} articles into ChromaDB",
                       agent="indexer", action="index_knowledge_base",
                       details={"article_count": len(articles)})
        return len(articles)

    finally:
        session.close()
