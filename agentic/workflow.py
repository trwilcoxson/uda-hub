"""UDA-Hub workflow entry point.

Provides the compiled orchestrator graph and a CLI for interactive support sessions.
"""
import argparse
import logging
import uuid

from langchain_core.messages import HumanMessage

from agentic.graph import app as orchestrator
from agentic.logging_config import setup_logging


def run_ticket(message: str, thread_id: str | None = None) -> str:
    """Process a single support message and return the response.

    Args:
        message: Customer support message.
        thread_id: Session identifier. Auto-generated if omitted.

    Returns:
        The final assistant response text.
    """
    thread_id = thread_id or str(uuid.uuid4())
    result = orchestrator.invoke(
        {"messages": [HumanMessage(content=message)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    return result["messages"][-1].content


def interactive(thread_id: str | None = None):
    """Launch an interactive chat session with the UDA-Hub support system."""
    thread_id = thread_id or str(uuid.uuid4())
    print(f"UDA-Hub Support  |  session {thread_id}")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input or user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        response = run_ticket(user_input, thread_id=thread_id)
        print(f"Agent: {response}\n")


def main():
    parser = argparse.ArgumentParser(description="UDA-Hub customer support system")
    sub = parser.add_subparsers(dest="command")

    chat_parser = sub.add_parser("chat", help="Interactive support chat")
    chat_parser.add_argument("--thread", help="Session/thread ID")

    run_parser = sub.add_parser("run", help="Process a single message")
    run_parser.add_argument("message", help="Customer message to process")
    run_parser.add_argument("--thread", help="Session/thread ID")

    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    setup_logging(level=logging.DEBUG if args.verbose else logging.INFO)

    if args.command == "run":
        response = run_ticket(args.message, thread_id=args.thread)
        print(response)
    elif args.command == "chat":
        interactive(thread_id=args.thread)
    else:
        interactive()


if __name__ == "__main__":
    main()
