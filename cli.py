"""
FoodieHelp AI — Interactive Terminal CLI
Chat with FoodieHelp customer support agent directly from your terminal.
"""

from __future__ import annotations

import sys
from agent.orchestrator import agent_orchestrator
from database.db import init_db, get_all_customers
from database.seed import seed_database


def run_cli():
    """Run an interactive CLI chat loop."""
    init_db()
    seed_database(force_reseed=False)

    customers = get_all_customers()
    current_customer_id = "CUST001"
    pending_confirmation = None

    print("\n" + "=" * 65)
    print("🍔  FOODIEHELP AI — CUSTOMER SUPPORT CONSOLE")
    print("=" * 65)
    print(f"Logged in as: {current_customer_id} (Type '/customers' to list or '/switch <id>' to change)")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            prompt = input(f"\n[{current_customer_id}] You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not prompt:
            continue

        if prompt.lower() in ["exit", "quit", "q"]:
            print("Session ended. Have a great day!")
            break

        if prompt.lower() == "/customers":
            print("\nAvailable Customers:")
            for c in customers[:8]:
                print(f"  • {c.customer_id}: {c.name} ({c.phone})")
            continue

        if prompt.lower().startswith("/switch"):
            parts = prompt.split()
            if len(parts) > 1:
                new_id = parts[1].strip().upper()
                current_customer_id = new_id
                pending_confirmation = None
                print(f"Switched active profile to: {current_customer_id}")
            else:
                print("Usage: /switch CUST002")
            continue

        # Process through agent orchestrator
        resp = agent_orchestrator.process_message(
            user_message=prompt,
            customer_id=current_customer_id,
            session_state={"pending_confirmation": pending_confirmation}
        )

        pending_confirmation = resp.pending_confirmation

        # Print Response
        print("-" * 65)
        if resp.action_taken and resp.action_taken != "None":
            print(f"⚡ [Action: {resp.action_taken}]")
            if resp.tool_used:
                print(f"🛠️  [Tool: {resp.tool_used}]")
            print()

        print(resp.content)

        if resp.sources:
            print("\n📚 Grounded Policy Sources:")
            for s in resp.sources:
                print(f"  • {s['source']} (Page {s['page']}) — Match: {s['similarity']*100:.1f}%")

        if resp.pending_confirmation:
            print("\n⚠️ [Confirmation Required: Please reply 'yes' or 'confirm' to proceed, or 'no' to keep order]")

        print("-" * 65)


if __name__ == "__main__":
    run_cli()

