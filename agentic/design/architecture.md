# UDA-Hub Architecture

## System Overview

UDA-Hub is a multi-agent customer support system for **CultPass**, a Brazilian cultural experiences subscription service. It uses LangGraph's Supervisor pattern to orchestrate specialized agents that classify, route, and resolve support tickets.

## Agent Architecture

### Supervisor (Router)
- **Role**: Orchestrates all routing; composes final response to customer
- **Model**: gpt-4o-mini
- **Pattern**: `langgraph_supervisor.create_supervisor()`
- **Routing rules**:
  1. New messages → Triage Agent (always first)
  2. General questions → Knowledge Agent
  3. Account inquiries → Account Agent
  4. Action requests → Action Agent
  5. Escalation → Compose escalation response

### Triage Agent
- **Role**: Classifies tickets by issue_type, priority, sentiment using LLM-powered analysis
- **Tools**: `classify_ticket` — invokes `gpt-4o-mini` with `with_structured_output(TicketClassification)` for reliable structured JSON
- **Output**: `TicketClassification` (Pydantic model) with `issue_type`, `priority`, `sentiment`, `requires_human`, `summary`
- **Classification method**: The LLM reads the full customer message and produces all fields in a single inference call — no keyword heuristics
- **Side effect**: Updates `TicketMetadata` in udahub.db

### Knowledge Agent (RAG)
- **Role**: Retrieves answers from the knowledge base
- **Tools**: `search_knowledge`, `get_article_by_id`
- **RAG Pipeline**: ChromaDB with OpenAI `text-embedding-3-small` embeddings
- **Confidence threshold**: 0.7 (below = flag for escalation)

### Account Agent
- **Role**: Read-only lookups against CultPass customer data + memory-aware personalization
- **Tools**: `lookup_user`, `get_subscription`, `get_reservations`, `get_customer_context`, `record_customer_preference`
- **Database**: cultpass.db (read-only), udahub.db (memory read/write)
- **Memory integration**: After identifying a customer, loads their past resolutions and preferences from long-term memory to personalize responses. Stores new preferences when mentioned.

### Action Agent
- **Role**: Performs write operations for support resolution + records outcomes
- **Tools**: `cancel_reservation`, `process_refund`, `update_subscription`, `record_resolution`
- **Database**: cultpass.db (read/write), udahub.db (resolution storage)
- **Memory integration**: After completing actions, records the resolution in long-term memory for future reference.

## State Schema

```python
class AgentState(MessagesState):
    next: str         # Next agent to route to
    ticket_id: str    # Current ticket ID
    user_email: str   # Current user email
```

## RAG Pipeline

1. **Indexing** (`agentic/rag/indexer.py`):
   - Loads Knowledge articles from udahub.db
   - Embeds with `text-embedding-3-small`
   - Stores in ChromaDB persistent collection

2. **Retrieval** (`agentic/rag/retriever.py`):
   - `KnowledgeRetriever.search(query, top_k=3)`
   - Confidence = 1 / (1 + L2_distance)
   - Threshold: 0.7

## Sentiment-Driven Routing

Sentiment analysis directly influences routing and response behavior:

- **Frustrated/Negative sentiment**: Prioritized handling, empathetic tone, escalation considered if combined with high/urgent priority
- **Positive sentiment**: Acknowledged in response to reinforce good experience
- **Sentiment + Priority matrix**: Determines whether ticket should be fast-tracked or escalated

The supervisor uses both `issue_type` and `sentiment` from triage classification to make routing decisions.

## Memory Strategy

- **Short-term**: LangGraph `MemorySaver` with `thread_id` — automatic conversation continuity within a session
- **Long-term** (3 layers in udahub.db):
  - `TicketMessage` table: Full conversation history — `save_message()` / `load_conversation_history()`
  - `TicketResolution` table: Resolved issue records — `save_resolution()` / `load_resolutions_for_user()` — enables learning from past resolutions across sessions
  - `CustomerPreference` table: Stored preferences (language, contact method, etc.) — `save_customer_preference()` / `load_customer_preferences()` — enables personalized support across sessions

## Databases

| Database | Purpose | Access |
|----------|---------|--------|
| `cultpass.db` | Customer data (users, subscriptions, experiences, reservations) | Account Agent (read), Action Agent (read/write) |
| `udahub.db` | Support system (accounts, tickets, messages, knowledge base, resolutions, preferences) | Triage (write metadata), Knowledge (read KB), Account (read/write memory), Action (write resolutions) |
| ChromaDB | Vector embeddings for RAG (`text-embedding-3-small`) | Knowledge Agent (read) |

## Escalation Logic

Tickets are escalated to human support when:
- `requires_human = true` from triage classification
- RAG confidence is below threshold (0.7)
- Customer explicitly requests human agent
- Frustrated sentiment + high/urgent priority
- Legal, compliance, or discrimination concerns detected

## Structured Logging

All agent actions emit structured JSON logs with consistent fields:
- `timestamp`: ISO 8601 UTC
- `level`: INFO, WARNING, ERROR
- `agent`: Which agent emitted the log
- `action`: What action was performed
- `ticket_id`: (when available) Associated ticket
- `details`: Structured payload with operation-specific data

Configured via `agentic/logging_config.py` using `StructuredFormatter`.

## Stand-out Features

### MCP Server
`agentic/tools/mcp_server.py` exposes account and action tools via FastMCP protocol, enabling integration with external MCP clients.

### Sentiment Analysis
Triage agent classifies sentiment (positive, neutral, negative, frustrated) which directly influences supervisor routing decisions and response tone.

### Advanced Knowledge Retrieval
RAG pipeline uses OpenAI `text-embedding-3-small` embeddings with ChromaDB vector store for semantic search with confidence scoring.

## Tool Inventory

| Tool | Agent | Purpose |
|------|-------|---------|
| `classify_ticket` | Triage | LLM-powered structured ticket classification (gpt-4o-mini + Pydantic) |
| `search_knowledge` | Knowledge | RAG search across KB articles |
| `get_article_by_id` | Knowledge | Direct article lookup |
| `lookup_user` | Account | Find user by email |
| `get_subscription` | Account | Get subscription details |
| `get_reservations` | Account | Get booking history |
| `get_customer_context` | Account | Load past resolutions + preferences from long-term memory |
| `record_customer_preference` | Account | Save customer preference for cross-session personalization |
| `cancel_reservation` | Action | Cancel a booking |
| `process_refund` | Action | Process refund for cancelled booking |
| `update_subscription` | Action | Pause or cancel subscription |
| `record_resolution` | Action | Store resolution in long-term memory for future reference |
