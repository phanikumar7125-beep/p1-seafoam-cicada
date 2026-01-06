import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_refund_flow():
    res = client.post(
        "/triage/invoke",
        json={"ticket_text": "Refund request for order ORD1001"}
    )

    assert res.status_code == 200
    body = res.json()

    assert body["order_id"] == "ORD1001"
    assert body["issue_type"] == "refund_request"
    assert "refund" in body["reply_text"].lower()
