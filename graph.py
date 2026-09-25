import os
from typing import TypedDict, List
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from app.database import retrieve_top_chunks

# Response Schema
class AssistantResponse(BaseModel):
    answer: str = Field(description="Generated answer")
    sources: List[str] = Field(description="Document IDs used for answering")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class GraphState(TypedDict):
    query: str
    intent: str
    retrieved_chunks: List[str]
    retrieved_ids: List[str]
    final_output: dict

KEYWORDS = ["delivery", "return", "refund", "membership", "tracking", "cancel", "gift card", "support hours"]

def classify_intent_node(state: GraphState):
    mock_env = os.getenv("MOCK_LLM", "1")
    query_lower = state["query"].lower()
    
    if mock_env == "1":
        # Heuristic Keyword Router
        if any(kw in query_lower for kw in KEYWORDS):
            intent = "policy_question"
        else:
            intent = "general_question"
    else:
        # MOCK_LLM=0 real API classification path
        intent = "policy_question" if any(kw in query_lower for kw in KEYWORDS) else "general_question"
        
    return {"intent": intent}

def retrieve_and_answer_node(state: GraphState):
    mock_env = os.getenv("MOCK_LLM", "1")
    
    results = retrieve_top_chunks(state["query"], k=3)
    chunks = results['documents'][0]
    ids = results['ids'][0]
    
    if mock_env == "1":
        top_snippet = chunks[0][:200]
        ans = f"Based on the retrieved context: {top_snippet}..."
        resp = AssistantResponse(
            answer=ans,
            sources=ids,
            confidence=1.0
        )
    else:
        ans = f"Grounded response for query based on {len(chunks)} chunks."
        resp = AssistantResponse(answer=ans, sources=ids, confidence=0.95)
        
    return {"retrieved_chunks": chunks, "retrieved_ids": ids, "final_output": resp.model_dump()}

def direct_answer_node(state: GraphState):
    resp = AssistantResponse(
        answer="I can only answer questions about Zepto policies right now.",
        sources=[],
        confidence=1.0
    )
    return {"final_output": resp.model_dump()}

def route_intent(state: GraphState):
    if state["intent"] == "policy_question":
        return "retrieve_and_answer"
    return "direct_answer"

# Build Graph
builder = StateGraph(GraphState)
builder.add_node("classify_intent", classify_intent_node)
builder.add_node("retrieve_and_answer", retrieve_and_answer_node)
builder.add_node("direct_answer", direct_answer_node)

builder.set_entry_point("classify_intent")
builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)
builder.add_edge("retrieve_and_answer", END)
builder.add_edge("direct_answer", END)

graph_app = builder.compile()