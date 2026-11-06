"""
FoodieHelp AI — Customer Support Web Platform
Provides a clean, abstracted customer support interface.
Rules, policies, and internal vector mechanics are securely abstracted on the backend.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from config.settings import settings
from database.db import init_db, get_order_by_id
from database.seed import seed_database
from ingestion.vector_store import VectorStore, index_directory
from agent.orchestrator import agent_orchestrator
from services.order_service import order_service


class ChatRequest(BaseModel):
    message: str
    customer_id: str = "CUST001"
    pending_confirmation: Optional[Dict[str, Any]] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize system database and verified internal knowledge base on startup."""
    init_db()
    seed_database(force_reseed=False)
    try:
        vstore = VectorStore()
        stats = vstore.get_stats()
        if stats["total_chunks"] == 0:
            index_directory(force_reindex=False)
    except Exception as e:
        print(f"[FoodieHelp] Knowledge base initialization: {e}")
    yield


app = FastAPI(
    title="FoodieHelp Support",
    description="Customer Support Platform for Food Delivery Orders and Policies.",
    version="2.1.0",
    lifespan=lifespan
)

templates_dir = Path(__file__).resolve().parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))


@app.get("/", response_class=HTMLResponse)
async def customer_support_home(request: Request):
    """Serve the FoodieHelp Customer Support Chat interface."""
    return templates.TemplateResponse(request=request, name="index.html")


@app.post("/api/chat")
async def chat_endpoint(req: ChatRequest):
    """
    Chat endpoint:
    Processes customer inquiries, order tracking, cancellations, and policy queries.
    """
    resp = agent_orchestrator.process_message(
        user_message=req.message,
        customer_id=req.customer_id,
        session_state={"pending_confirmation": req.pending_confirmation}
    )
    return {
        "content": resp.content,
        "action_taken": resp.action_taken,
        "tool_used": resp.tool_used,
        "tool_data": resp.tool_data,
        "sources": resp.sources,
        "pending_confirmation": resp.pending_confirmation,
        "suggested_replies": resp.suggested_replies
    }


@app.get("/api/orders/{order_id}")
async def lookup_order(order_id: str):
    """Look up order details by Order ID."""
    clean_id = order_id.strip().upper()
    order = get_order_by_id(clean_id)
    if not order:
        return {"found": False, "message": f"Order {clean_id} not found."}
    return {
        "found": True,
        "order_id": order.order_id,
        "restaurant_name": order.restaurant_name,
        "status": order.order_status,
        "amount": order.order_amount,
        "estimated_delivery": order.estimated_delivery_time
    }


if __name__ == "__main__":
    print("\n🍔 Starting FoodieHelp Customer Support on http://127.0.0.1:8000 ...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
