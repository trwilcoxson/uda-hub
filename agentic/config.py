"""Central configuration for UDA-Hub: paths, model settings, RAG thresholds."""
from pathlib import Path

# Base directory (solution/)
BASE_DIR = Path(__file__).resolve().parent.parent

# Database paths
CULTPASS_DB_PATH = str(BASE_DIR / "data" / "external" / "cultpass.db")
UDAHUB_DB_PATH = str(BASE_DIR / "data" / "core" / "udahub.db")

# ChromaDB
CHROMA_PERSIST_DIR = str(BASE_DIR / "data" / "core" / "chromadb")
CHROMA_COLLECTION_NAME = "cultpass_knowledge"

# Model configuration
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# RAG configuration
RAG_CONFIDENCE_THRESHOLD = 0.7
RAG_TOP_K = 3

# Account
ACCOUNT_ID = "cultpass"
