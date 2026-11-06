"""
Integration Tests for FoodieHelp AI FastAPI Application
Tests web chat interface and API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app import app
from database.seed import seed_database


@pytest.fixture(scope="module", autouse=True)
def init_data():
    seed_database(force_reseed=False)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_home_page_renders(client):
    """Test that the Customer Support Chat page loads with 200 OK."""
    response = client.get("/")
    assert response.status_code == 200
    assert "FoodieHelp Support" in response.text


def test_api_chat_endpoint(client):
    """Test chatting with the agent via REST API."""
    payload = {
        "message": "Where is my order ORD1001?",
        "customer_id": "CUST001"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "content" in data
    assert "action_taken" in data
    assert len(data["content"]) > 0


def test_api_order_lookup(client):
    """Test direct order lookup endpoint."""
    response = client.get("/api/orders/ORD1001")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is True
    assert data["order_id"] == "ORD1001"


def test_api_order_lookup_not_found(client):
    """Test order lookup for non-existent order returns found=False."""
    response = client.get("/api/orders/ORD9999")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False
    assert "not found" in data["message"].lower()


def test_api_chat_human_escalation(client):
    """Test escalating to human support executive via REST API."""
    payload = {
        "message": "I want to speak with a human agent",
        "customer_id": "CUST001"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "Human Support" in data["action_taken"]
    assert "TIK-" in data["content"]


def test_api_chat_out_of_scope_query(client):
    """Test that external questions like 'What is metals' are politely declined."""
    payload = {
        "message": "What is metals",
        "customer_id": "CUST001"
    }
    response = client.post("/api/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    assert len(data["content"]) > 0
    # Must mention FoodieHelp or food support focus
    assert any(w in data["content"].lower() for w in ["food", "foodiehelp", "restaurant", "order", "refund"])
