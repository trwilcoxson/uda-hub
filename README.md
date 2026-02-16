# UDA-Hub: Multi-Agent Customer Support for CultPass

A multi-agent customer support system built with LangGraph's Supervisor pattern. UDA-Hub orchestrates specialized AI agents to classify, route, and resolve support tickets for CultPass, a Brazilian cultural experiences subscription service.

## Architecture

- **Supervisor**: Routes messages to specialized agents and composes responses
- **Triage Agent**: Classifies tickets by issue type, priority, and sentiment
- **Knowledge Agent**: RAG-powered search across 18 knowledge base articles
- **Account Agent**: Read-only customer data lookups
- **Action Agent**: Write operations (cancel reservations, refunds, subscription changes)

## Setup

### Requirements

- Python 3.11+
- OpenAI API key

### Installation

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Database Setup

Run the notebooks in order from the `solution/` directory:

1. **`01_external_db_setup.ipynb`** - Creates `cultpass.db` with users, experiences, subscriptions, reservations
2. **`02_core_db_setup.ipynb`** - Creates `udahub.db` with knowledge base, tickets, and indexes articles into ChromaDB

### Running

```bash
# Run the demo notebook
jupyter notebook 03_agentic_app.ipynb

# Or use the interactive chat
python -c "from utils import chat_interface; from workflow import orchestrator; chat_interface(orchestrator, '1')"
```

### Testing

```bash
# Run unit tests (no API key needed for mocked tests)
pytest tests/test_triage.py tests/test_knowledge.py -v

# Run integration tests (requires databases)
pytest tests/test_tools.py tests/test_memory.py -v

# Run all tests including e2e (requires databases + API key)
pytest tests/ -v
```

### MCP Server

```bash
# Run the FastMCP server for external tool integration
python -m agentic.tools.mcp_server
```

## Project Structure

```
solution/
├── agentic/                    # Core agent system
│   ├── config.py               # Configuration (DB paths, model, RAG settings)
│   ├── state.py                # LangGraph state schema
│   ├── graph.py                # Supervisor graph assembly
│   ├── workflow.py             # Entry point
│   ├── agents/                 # Agent definitions
│   ├── tools/                  # Tool implementations + MCP server
│   ├── rag/                    # RAG pipeline (indexer + retriever)
│   ├── memory/                 # Long-term persistence
│   └── design/                 # Architecture docs + diagrams
├── data/                       # Databases and data files
├── tests/                      # Test suite
├── 01_external_db_setup.ipynb  # CultPass DB setup
├── 02_core_db_setup.ipynb      # UDA-Hub DB + RAG indexing
├── 03_agentic_app.ipynb        # Demo notebook
├── utils.py                    # Shared utilities
└── workflow.py                 # Top-level import
```
