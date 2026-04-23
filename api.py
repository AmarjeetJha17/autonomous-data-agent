# api.py
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from langchain_core.messages import HumanMessage, AIMessage
from agent import app as agent_app  # Import your compiled LangGraph app

app = FastAPI(title="Autonomous Data Agent API")
AGENT_TIMEOUT_SECONDS = 180

class Message(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    messages: List[Message]

class QueryResponse(BaseModel):
    answer: str


def run_agent(langchain_messages):
    events = agent_app.stream(
        {"messages": langchain_messages},
        stream_mode="values",
        config={"recursion_limit": 20},
    )

    final_response = ""
    all_messages = []
    
    for event in events:
        if "messages" in event:
            all_messages = event["messages"]

    # Find the last user-facing message (not SystemMessage, not ToolMessage)
    for msg in reversed(all_messages):
        msg_type = type(msg).__name__
        
        # Skip system and tool messages
        if msg_type in ("SystemMessage", "ToolMessage", "ToolCall", "ToolCallBlock"):
            continue
        
        # Get any content from AIMessage or HumanMessage
        if hasattr(msg, 'content') and msg.content:
            content = msg.content
            if isinstance(content, str) and content.strip():
                final_response = content
                break

    return final_response

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        # Convert the dictionary messages to LangChain message objects
        langchain_messages = []
        for msg in request.messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
                
        # Run agent execution in a worker thread and enforce a hard timeout.
        final_response = await asyncio.wait_for(
            asyncio.to_thread(run_agent, langchain_messages),
            timeout=AGENT_TIMEOUT_SECONDS,
        )
                
        return QueryResponse(answer=final_response)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=f"The analysis timed out after {AGENT_TIMEOUT_SECONDS} seconds. Please try a simpler query.",
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)