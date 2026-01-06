
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import TypedDict, Optional, List
import json, os, re

# LangGraph
from langgraph.graph import StateGraph



app = FastAPI(title="Phase 1 Mock API (LangGraph Agent)")

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MOCK_DIR = os.path.join(ROOT, "mock_data")

def load(name):
    with open(os.path.join(MOCK_DIR, name), "r", encoding="utf-8") as f:
        return json.load(f)

ORDERS = load("orders.json")
ISSUES = load("issues.json")
REPLIES = load("replies.json")



class TriageInput(BaseModel):
    ticket_text: str
    order_id: Optional[str] = None


class TriageState(TypedDict):
    messages: List[str]
    ticket_text: str
    order_id: Optional[str]
    issue_type: Optional[str]
    evidence: Optional[dict]
    recommendation: Optional[str]



def ingest(state: TriageState) -> TriageState:
    # Extract order_id if missing
    if not state["order_id"]:
        match = re.search(r"(ORD\d{4})", state["ticket_text"], re.IGNORECASE)
        if match:
            state["order_id"] = match.group(1).upper()

    if not state["order_id"]:
        raise HTTPException(status_code=400, detail="order_id missing and not found in text")

    state["messages"].append("Ingested input and resolved order_id")
    return state


def classify_issue_node(state: TriageState) -> TriageState:
    text = state["ticket_text"].lower()

    for rule in ISSUES:
        if rule["keyword"] in text:
            state["issue_type"] = rule["issue_type"]
            state["messages"].append(f"Issue classified as {rule['issue_type']}")
            return state

    state["issue_type"] = "unknown"
    state["messages"].append("Issue classified as unknown")
    return state


def fetch_order_node(state: TriageState) -> TriageState:
    order = next((o for o in ORDERS if o["order_id"] == state["order_id"]), None)
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    state["evidence"] = order
    state["messages"].append(f"Fetched order {state['order_id']}")
    return state


def draft_reply_node(state: TriageState) -> TriageState:
    template = next(
        (r["template"] for r in REPLIES if r["issue_type"] == state["issue_type"]),
        "Hi {{customer_name}}, we are reviewing order {{order_id}}."
    )

    order = state["evidence"]
    reply = (
        template
        .replace("{{customer_name}}", order.get("customer_name", "Customer"))
        .replace("{{order_id}}", order.get("order_id", ""))
    )

    state["recommendation"] = reply
    state["messages"].append("Drafted customer reply")
    return state


graph = StateGraph(TriageState)

graph.add_node("ingest", ingest)
graph.add_node("classify_issue", classify_issue_node)
graph.add_node("fetch_order", fetch_order_node)
graph.add_node("draft_reply", draft_reply_node)

graph.set_entry_point("ingest")
graph.add_edge("ingest", "classify_issue")
graph.add_edge("classify_issue", "fetch_order")
graph.add_edge("fetch_order", "draft_reply")

triage_graph = graph.compile()



@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/orders/get")
def orders_get(order_id: str = Query(...)):
    for o in ORDERS:
        if o["order_id"] == order_id:
            return o
    raise HTTPException(status_code=404, detail="Order not found")

@app.get("/orders/search")
def orders_search(customer_email: Optional[str] = None, q: Optional[str] = None):
    matches = []
    for o in ORDERS:
        if customer_email and o["email"].lower() == customer_email.lower():
            matches.append(o)
        elif q and (
            o["order_id"].lower() in q.lower()
            or o["customer_name"].lower() in q.lower()
        ):
            matches.append(o)
    return {"results": matches}



@app.post("/triage/invoke")
def triage_invoke(body: TriageInput):
    initial_state: TriageState = {
        "messages": [],
        "ticket_text": body.ticket_text,
        "order_id": body.order_id,
        "issue_type": None,
        "evidence": None,
        "recommendation": None,
    }

    final_state = triage_graph.invoke(initial_state)

    return {
        "order_id": final_state["order_id"],
        "issue_type": final_state["issue_type"],
        "order": final_state["evidence"],
        "reply_text": final_state["recommendation"],
        "trace": final_state["messages"],
    }
