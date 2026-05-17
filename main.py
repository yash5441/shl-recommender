"""
main.py - FastAPI service exposing /health and /chat endpoints.
"""

import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from models import ChatRequest, ChatResponse, HealthResponse
from agent import run_agent
from retriever import retriever

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shl-recommender")

# ---------------------------------------------------------------------------
# Startup: load catalog into retriever
# ---------------------------------------------------------------------------
startup_ok = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    global startup_ok
    logger.info("Starting SHL Recommender service...")
    t0 = time.time()

    ok = retriever.load()
    if ok:
        logger.info(
            f"Catalog loaded: {retriever.count()} assessments in {time.time()-t0:.1f}s"
        )
        startup_ok = True
    else:
        logger.warning(
            "Catalog failed to load. Service will start but recommendations will be empty."
        )
        startup_ok = False

    yield  # service is running

    logger.info("Shutting down SHL Recommender service.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for SHL assessment selection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Middleware: request logging + timeout guard
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    t0 = time.time()
    response = await call_next(request)
    elapsed = time.time() - t0
    logger.info(
        f"{request.method} {request.url.path} -> {response.status_code} ({elapsed:.2f}s)"
    )
    return response


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse, status_code=200)
async def health():
    """Readiness check. Returns 200 when service is up."""
    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse, status_code=200)
async def chat(request: ChatRequest):
    """
    Stateless conversational endpoint.
    Accepts full message history, returns next agent reply + optional shortlist.
    """
    messages = request.messages

    # Basic validation
    if not messages:
        raise HTTPException(status_code=422, detail="messages list cannot be empty.")

    # Validate roles
    for msg in messages:
        if msg.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=422,
                detail=f"Invalid role '{msg.role}'. Must be 'user' or 'assistant'.",
            )

    # Ensure last message is from user
    if messages[-1].role != "user":
        raise HTTPException(
            status_code=422,
            detail="Last message must be from 'user'.",
        )

    # Turn cap: 8 turns max (user + assistant combined)
    if len(messages) > 8:
        return ChatResponse(
            reply="This conversation has reached the maximum length. Please start a new session.",
            recommendations=[],
            end_of_conversation=True,
        )

    try:
        response = run_agent(messages)
        return response
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Internal error in agent. Please retry.",
        )


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
