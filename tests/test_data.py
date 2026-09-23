"""Data-generation tests: determinism, schema, split sanity."""

import hashlib
import json
import os

import data_gen
import dataset


def _hashes(d):
    out = {}
    for name in ("jobs.json", "resumes.json", "split.json"):
        with open(os.path.join(d, name), "rb") as f:
            out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def test_generate_deterministic_in_memory():
    a = data_gen.generate(seed=42)
    b = data_gen.generate(seed=42)
    assert json.dumps(a) == json.dumps(b)


def test_generate_different_seed_differs():
    a = data_gen.generate(seed=42)
    b = data_gen.generate(seed=7)
    assert json.dumps(a[0]) != json.dumps(b[0])


def test_main_writes_identical_files_twice(tmp_path):
    d1, d2 = tmp_path / "a", tmp_path / "b"
    d1.mkdir()
    d2.mkdir()
    old = data_gen.DATA_DIR
    try:
        data_gen.DATA_DIR = str(d1)
        data_gen.main()
        data_gen.DATA_DIR = str(d2)
        data_gen.main()
    finally:
        data_gen.DATA_DIR = old
    assert _hashes(d1) == _hashes(d2)


def test_schema_and_counts(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    old = data_gen.DATA_DIR
    try:
        data_gen.DATA_DIR = str(d)
        data_gen.main()
    finally:
        data_gen.DATA_DIR = old
    jobs = json.load(open(d / "jobs.json"))
    resumes = json.load(open(d / "resumes.json"))
    split = json.load(open(d / "split.json"))
    meta = json.load(open(d / "meta.json"))

    assert len(jobs) == 12 * data_gen.N_JOBS_PER_FAMILY
    assert len(resumes) == 12 * data_gen.N_RESUMES_PER_FAMILY
    n_jobs = 12 * data_gen.N_JOBS_PER_FAMILY
    assert len(split["train_job_ids"]) == int(n_jobs * (1 - data_gen.TEST_FRAC))
    assert len(split["test_job_ids"]) == n_jobs - int(n_jobs * (1 - data_gen.TEST_FRAC))
    # every family represented in both splits
    by_id = {j["job_id"]: j["family"] for j in jobs}
    assert {by_id[i] for i in split["test_job_ids"]} == set(data_gen.FAMILIES)
    assert {by_id[i] for i in split["train_job_ids"]} == set(data_gen.FAMILIES)
    # no overlap, full coverage
    assert not set(split["train_job_ids"]) & set(split["test_job_ids"])
    assert len(set(split["train_job_ids"]) | set(split["test_job_ids"])) == n_jobs
    # required fields + non-empty text
    for j in jobs:
        assert j["text"] and j["family"] in data_gen.FAMILIES
    for r in resumes:
        assert r["text"] and r["family"] in data_gen.FAMILIES
        assert r["candidate_id"] >= 0
    assert "synthetic" in meta["disclaimer"].lower()


def test_no_real_names_or_companies():
    # spot-check the templates: no company names, no person names anywhere
    jobs, resumes, _ = data_gen.generate(seed=123)
    blob = " ".join(j["text"] for j in jobs[:20])
    blob += " ".join(r["text"] for r in resumes[:20])
    for token in ("Inc", "LLC", "Corp", "Google", "Amazon", "Microsoft"):
        assert token not in blob
    # candidate labels are synthetic ids, not names
    assert resumes[0]["text"].startswith("Candidate 0000")
