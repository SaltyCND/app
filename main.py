#!/usr/bin/env python
import asyncio
import os
import re  # For spacing fixes
import io
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent_manager import AgentManager
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi.responses import StreamingResponse, JSONResponse

app = FastAPI(
    title="Electrical AI App API",
    description="A multi-agent AI application for electrical industry tasks including technical Q&A, report generation, receipt processing, and more.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

agent_manager = AgentManager()

# Data models for API endpoints
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

# Streaming endpoint for chatbot (unchanged)
@app.get("/query/stream")
async def stream_query(query: str = Query(...)):
    try:
        final_answer = await agent_manager.chatbot.generateAnswer(query)
        async def answer_stream():
            processed_answer = re.sub(r'([a-z])([A-Z])', r'\1 \2', final_answer)
            print("Processed answer to stream:", processed_answer)
            words = processed_answer.split()
            for word in words:
                yield f"data: {word}\n\n"
                await asyncio.sleep(0.1)
            yield "data: [DONE]\n\n"
        headers = {"Cache-Control": "no-cache"}
        return StreamingResponse(answer_stream(), media_type="text/event-stream", headers=headers)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Non-streaming endpoint for CLI usage (if needed)
@app.post("/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    try:
        answer = await agent_manager.process_interactive_query(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Report Generation
@app.post("/report", response_model=QueryResponse)
async def report_endpoint(request: QueryRequest):
    try:
        report = await agent_manager.report_agent.generate_report(request.query)
        return QueryResponse(answer=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Training Quiz Generation
@app.post("/quiz", response_model=QueryResponse)
async def quiz_endpoint(request: QueryRequest):
    try:
        quiz = await agent_manager.training_agent.generate_quiz(request.query)
        return QueryResponse(answer=quiz)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for Diagram Analysis (file upload)
@app.post("/diagram")
async def diagram_endpoint(file: UploadFile = File(...)):
    try:
        content = await file.read()
        analysis = await agent_manager.diagram_agent.analyze_diagram(content)
        return JSONResponse(content={"result": analysis})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# (Receipt upload endpoint already exists in your project.)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)