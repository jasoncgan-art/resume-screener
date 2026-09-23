import json
import os

import pytest
from fastapi.testclient import TestClient

from serving.app import app
from serving.config import load_settings

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
with open(os.path.join(REPO_DIR, "data", "split.json"), encoding="utf-8") as f:
    TEST_JOB_IDS = json.load(f)["test_job_ids"]
with open(os.path.join(REPO_DIR, "data", "jobs.json"), encoding="utf-8") as f:
    N_JOBS = len(json.load(f))
with open(os.path.join(REPO_DIR, "data", "resumes.json"), encoding="utf-8") as f:
    N_RESUMES = len(json.load(f))


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "bi-encoder"
    assert body["n_resumes"] == N_RESUMES
    assert body["n_jobs"] == N_JOBS
    assert body["uptime_s"] >= 0


def test_screen_by_job_id(client):
    jid = TEST_JOB_IDS[0]
    r = client.post("/screen", json={"job_id": jid, "k": 10})
    assert r.status_code == 200
    body = r.json()
    assert body["k"] == 10
    items = body["items"]
    assert len(items) == 10
    assert [it["rank"] for it in items] == list(range(1, 11))
    scores = [it["score"] for it in items]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
    assert len({it["candidate_id"] for it in items}) == 10
    assert all(isinstance(it["candidate_id"], int) for it in items)
    assert body["latency_ms"] > 0


def test_screen_by_job_text(client):
    r = client.post("/screen", json={
        "job_text": "Senior Data Scientist. Required skills: Python, statistics, "
                    "A/B testing, machine learning. Nice to have: SQL.",
        "k": 5,
    })
    assert r.status_code == 200
    assert len(r.json()["items"]) == 5


def test_screen_default_k(client):
    settings = load_settings()
    r = client.post("/screen", json={"job_id": TEST_JOB_IDS[1]})
    assert r.status_code == 200
    assert r.json()["k"] == settings.top_k_default


def test_screen_validation(client):
    # neither job_id nor job_text
    assert client.post("/screen", json={"k": 10}).status_code == 422
    # both provided
    assert client.post("/screen", json={"job_id": 1, "job_text": "abc"}).status_code == 422
    # k out of range
    assert client.post("/screen", json={"job_id": 1, "k": 0}).status_code == 422
    assert client.post("/screen", json={"job_id": 1, "k": 1000}).status_code == 422
    # empty job_text
    assert client.post("/screen", json={"job_text": "x"}).status_code == 422


def test_screen_unknown_job_id(client):
    r = client.post("/screen", json={"job_id": 999999, "k": 10})
    assert r.status_code == 404


def test_screen_determinism(client):
    payload = {"job_id": TEST_JOB_IDS[3], "k": 20}
    first = client.post("/screen", json=payload).json()
    second = client.post("/screen", json=payload).json()
    assert first["items"] == second["items"]


def test_screen_top_hit_is_relevant(client):
    # the serving corpus is synthetic with known families: the top hit for a
    # test JD should usually share the JD's family (sanity, not a guarantee
    # for every JD -- assert over a batch instead).
    from serving.app import screener as sc
    hits = 0
    for jid in TEST_JOB_IDS[:20]:
        fam = sc.data["job_family"][jid]
        top = client.post("/screen", json={"job_id": jid, "k": 1}).json()["items"][0]
        ridx = next(i for i, r in enumerate(sc.data["resumes"])
                    if r["candidate_id"] == top["candidate_id"])
        hits += sc.data["resume_family"][ridx] == fam
    assert hits >= 15, f"only {hits}/20 top-1 hits were same-family"


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["screen_requests"] >= 1
    assert body["screen_mean_latency_ms"] is not None
    assert body["screen_mean_latency_ms"] > 0
