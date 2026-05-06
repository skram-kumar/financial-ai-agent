"""
Valura AI — FastAPI microservice with SSE streaming.

Pipeline per request:
  1. Safety Guard   — pure local, blocks harmful queries < 10ms
  2. Classifier     — 1 LLM call → intent + entities + agent
  3. Router         — dispatches to agent or stub
  4. SSE Stream     — streams response back to client token by token

Endpoints:
  POST /query       — main pipeline endpoint, streams SSE
  GET  /health      — health check
"""

from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import asyncio
import json
import logging
import os
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.safety import check as safety_check
from src.classifier import classify
from src.router import route

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Valura AI Microservice",
    description="AI co-investor for every user — build, monitor, grow, protect.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pipeline timeout in seconds — defend this in README
PIPELINE_TIMEOUT = 30

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    user: dict = {}               # user profile (portfolio, KYC, risk profile)
    prior_turns: list[str] = []   # conversation history for follow-up resolution
    session_id: str | None = None


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse_event(data: dict | str, event: str = "message") -> str:
    """Format a single SSE event."""
    if isinstance(data, dict):
        payload = json.dumps(data)
    else:
        payload = data
    return f"event: {event}\ndata: {payload}\n\n"


def _sse_error(message: str, code: str = "error") -> str:
    """Format an SSE error event."""
    return _sse_event({"error": message, "code": code}, event="error")


async def _stream_text(text: str, chunk_size: int = 8) -> AsyncGenerator[str, None]:
    """
    Stream text in small chunks to simulate token-by-token output.
    In production this would hook into the LLM's streaming API.
    """
    words = text.split(" ")
    buffer = []
    for word in words:
        buffer.append(word)
        if len(buffer) >= chunk_size:
            chunk = " ".join(buffer) + " "
            yield _sse_event({"type": "text_chunk", "content": chunk})
            buffer = []
            await asyncio.sleep(0.02)  # small delay for natural streaming feel
    if buffer:
        yield _sse_event({"type": "text_chunk", "content": " ".join(buffer)})


async def _run_pipeline(
    request: QueryRequest,
    llm_client: OpenAI,
) -> AsyncGenerator[str, None]:
    """
    Full pipeline: safety → classifier → router → stream response.
    Yields SSE events.
    """
    start_time = time.perf_counter()

    # ── 1. Safety Guard ──────────────────────────────────────────────────────
    yield _sse_event({"type": "status", "stage": "safety_check"})

    safety_verdict = safety_check(request.query)

    if safety_verdict.blocked:
        yield _sse_event({
            "type":     "blocked",
            "category": safety_verdict.category,
            "message":  safety_verdict.message,
        }, event="blocked")
        yield _sse_event({"type": "done", "blocked": True}, event="done")
        return

    # ── 2. Intent Classifier ─────────────────────────────────────────────────
    yield _sse_event({"type": "status", "stage": "classifying"})

    classifier_result = classify(
        query=request.query,
        llm=llm_client,
        prior_turns=request.prior_turns,
    )

    yield _sse_event({
        "type":           "classified",
        "agent":          classifier_result.agent,
        "intent":         classifier_result.intent,
        "entities":       classifier_result.entities,
        "safety_verdict": classifier_result.safety_verdict,
        "confidence":     classifier_result.confidence,
    }, event="classified")

    # ── 3. Router → Agent ────────────────────────────────────────────────────
    yield _sse_event({"type": "status", "stage": "generating"})

    agent_response = route(
        result=classifier_result,
        user=request.user,
        llm=llm_client,
    )

    # ── 4. Stream the response ───────────────────────────────────────────────

    # Stream narrative text if present (portfolio health agent)
    narrative = agent_response.get("narrative", "")
    if narrative:
        async for chunk in _stream_text(narrative):
            yield chunk

    # Stream structured data as a single event
    # Remove narrative from structured payload (already streamed as text)
    structured = {k: v for k, v in agent_response.items()
                  if k not in ("narrative", "_meta")}

    yield _sse_event({
        "type":     "structured_response",
        "agent":    classifier_result.agent,
        "response": structured,
    }, event="response")

    # Stream metadata
    elapsed = time.perf_counter() - start_time
    yield _sse_event({
        "type":        "done",
        "blocked":     False,
        "agent":       classifier_result.agent,
        "latency_ms":  round(elapsed * 1000, 1),
        "fallback":    classifier_result.fallback,
    }, event="done")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": "valura-ai",
        "version": "1.0.0",
    }


@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Main pipeline endpoint. Streams SSE events.

    Event types:
      status           — pipeline stage indicator
      blocked          — safety guard blocked the query
      classified       — classifier result (agent, entities, confidence)
      text_chunk       — streaming text chunk from agent narrative
      response         — full structured agent response
      done             — pipeline complete
      error            — error occurred

    Pipeline timeout: 30s (configurable via PIPELINE_TIMEOUT).
    """
    # Validate query
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Build LLM client
    llm_client = None  # classifier handles key selection internally

    async def generate():
        try:
            async with asyncio.timeout(PIPELINE_TIMEOUT):
                async for event in _run_pipeline(request, llm_client):
                    yield event
        except asyncio.TimeoutError:
            yield _sse_error(
                f"Request timed out after {PIPELINE_TIMEOUT}s.",
                code="timeout",
            )
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            yield _sse_error(str(exc), code="internal_error")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":               "no-cache",
            "X-Accel-Buffering":           "no",   # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )