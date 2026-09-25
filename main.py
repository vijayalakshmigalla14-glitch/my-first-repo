from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.database import initialize_vector_db
from app.graph import graph_app, AssistantResponse

app = FastAPI(title="Zepto GenAI Support Assistant")

@app.on_event("startup")
def startup_event():
    initialize_vector_db()

class QueryRequest(BaseModel):
    query: str

@app.post("/ask", response_model=AssistantResponse)
def ask_question(request: QueryRequest):
    try:
        initial_state = {
            "query": request.query,
            "intent": "",
            "retrieved_chunks": [],
            "retrieved_ids": [],
            "final_output": {}
        }
        res = graph_app.invoke(initial_state)
        return res["final_output"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))