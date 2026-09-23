"""FastAPI serving app for the resume-screener bi-encoder.

Endpoints:
    GET  /health     -> service + model status
    GET  /metrics    -> basic request counters
    POST /screen     -> ranked (candidate_id, score) list for a job description

Run:  uvicorn serving.app:app --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from serving.config import load_settings  # noqa: E402
from serving.model_loader import load_screener  # noqa: E402

settings = load_settings()
screener = None

_start_time = time.time()
_counters = {
    "health_requests": 0,
    "screen_requests": 0,
    "screen_errors": 0,
    "screen_latency_sum_s": 0.0,
}
_lock = threading.Lock()


class ScreenRequest(BaseModel):
    job_id: int | None = Field(
        default=None, ge=0,
        description="Corpus JD id to screen for (mutually exclusive with job_text)",
    )
    job_text: str | None = Field(
        default=None, min_length=3, max_length=20000,
        description="Raw job description text (mutually exclusive with job_id)",
    )
    k: int | None = Field(
        default=None,
        ge=1,
        le=settings.top_k_max,
        description=f"Number of candidates to return (1..{settings.top_k_max})",
    )


class ScreenItem(BaseModel):
    rank: int
    candidate_id: int
    title: str | None
    score: float


class ScreenResponse(BaseModel):
    k: int
    items: list[ScreenItem]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    n_resumes: int
    n_jobs: int
    uptime_s: float


class MetricsResponse(BaseModel):
    uptime_s: float
    health_requests: int
    screen_requests: int
    screen_errors: int
    screen_mean_latency_ms: float | None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global screener
    screener = load_screener(settings)  # model + resume index loaded once
    yield
    screener = None


app = FastAPI(title="resume-screener", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health():
    with _lock:
        _counters["health_requests"] += 1
    return HealthResponse(
        status="ok",
        model="bi-encoder",
        n_resumes=screener.n_resumes,
        n_jobs=len(screener.data["jobs"]),
        uptime_s=time.time() - _start_time,
    )


@app.get("/metrics", response_model=MetricsResponse)
def metrics():
    with _lock:
        c = dict(_counters)
    mean_ms = (c["screen_latency_sum_s"] / c["screen_requests"] * 1000
               if c["screen_requests"] else None)
    return MetricsResponse(
        uptime_s=time.time() - _start_time,
        health_requests=c["health_requests"],
        screen_requests=c["screen_requests"],
        screen_errors=c["screen_errors"],
        screen_mean_latency_ms=mean_ms,
    )


@app.post("/screen", response_model=ScreenResponse)
def screen(req: ScreenRequest):
    k = req.k if req.k is not None else settings.top_k_default
    t0 = time.perf_counter()
    try:
        if (req.job_id is None) == (req.job_text is None):
            raise HTTPException(
                status_code=422,
                detail="provide exactly one of job_id or job_text",
            )
        try:
            qvec = (screener.job_vector_by_id(req.job_id)
                    if req.job_id is not None
                    else screener.job_vector(req.job_text))
        except (KeyError, ValueError) as e:
            raise HTTPException(status_code=404, detail=str(e))
        results = screener.screen(qvec, k)
    except HTTPException:
        with _lock:
            _counters["screen_errors"] += 1
        raise
    latency_ms = (time.perf_counter() - t0) * 1000
    with _lock:
        _counters["screen_requests"] += 1
        _counters["screen_latency_sum_s"] += latency_ms / 1000
    items = []
    for rank, ridx, s in results:
        r = screener.data["resumes"][ridx]
        items.append(ScreenItem(rank=rank, candidate_id=r["candidate_id"],
                                title=r["title"], score=s))
    return ScreenResponse(k=k, items=items, latency_ms=latency_ms)
