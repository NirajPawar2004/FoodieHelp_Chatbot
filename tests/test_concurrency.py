"""
Concurrency and Thread Safety Tests for FoodieHelp AI
Tests concurrent execution of database queries, tool lookups, and API requests
to verify thread safety, absence of race conditions, and system stability under parallel load.
"""

import pytest
import concurrent.futures
from fastapi.testclient import TestClient
from app import app
from database.seed import seed_database
from database.db import get_order_by_id, get_customer_by_id
from agent.orchestrator import agent_orchestrator


@pytest.fixture(scope="module", autouse=True)
def setup_data():
    seed_database(force_reseed=False)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_concurrent_database_reads():
    """Verify multiple threads reading orders and customers simultaneously without locks or errors."""
    def read_task(order_id: str):
        order = get_order_by_id(order_id)
        assert order is not None
        assert order.order_id == order_id
        return True

    order_ids = ["ORD1001", "ORD1002", "ORD1003", "ORD1004", "ORD1005"] * 4
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(read_task, order_ids))

    assert all(results)
    assert len(results) == 20


def test_concurrent_orchestrator_requests():
    """Verify orchestrator processes multiple user messages concurrently without state corruption."""
    def chat_task(item):
        customer_id, message = item
        resp = agent_orchestrator.process_message(
            user_message=message,
            customer_id=customer_id
        )
        assert resp is not None
        assert len(resp.content) > 0
        return True

    tasks = [
        ("CUST001", "Where is my order ORD1001?"),
        ("CUST002", "What is your refund policy?"),
        ("CUST003", "Can I cancel my order?"),
        ("CUST004", "What if tamper seal is broken?"),
        ("CUST005", "Speak to a human"),
    ] * 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(chat_task, tasks))

    assert all(results)
    assert len(results) == 10


def test_concurrent_api_chat_requests(client):
    """Verify FastAPI chat endpoint handles concurrent HTTP requests cleanly."""
    def api_task(order_id):
        payload = {
            "message": f"Status of {order_id}",
            "customer_id": "CUST001"
        }
        resp = client.post("/api/chat", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "content" in data
        assert data["sources"] == []
        return True

    order_ids = ["ORD1001", "ORD1002", "ORD1003"] * 3
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(api_task, order_ids))

    assert all(results)
    assert len(results) == 9

