# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from agent import app as agent_app  # Import your compiled LangGraph app

app = FastAPI(title="Autonomous Data Agent API")

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        # Pass the user query to the LangGraph agent
        events = agent_app.stream(
            {"messages": [HumanMessage(content=request.query)]},
            stream_mode="values"
        )
        
        # Iterate through the stream to get the final AI response
        final_response = ""
        for event in events:
            if "messages" in event:
                final_response = event["messages"][-1].content
                
        return QueryResponse(answer=final_response)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)