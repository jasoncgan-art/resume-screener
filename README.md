# Resume Retrieval Prototype and ATS Benchmark Lab

An offline, AI-assisted learning and prototype project for investigating candidate-job
ranking quality. It combines a BM25 baseline, a trainable PyTorch bi-encoder,
validation-selected reciprocal-rank fusion and a separate ATS comparison toolkit.

**Status:** the bundled corpus and supervision are synthetic. No ATS rankings,
hiring-manager judgments, real users or company production outcomes have been
measured here. The business motivation is an HR request for a better prototype;
that request is not evidence of a successful pilot.

## Reproduce

Python 3.12 was used for the recorded run. CPU is sufficient.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r serving/requirements.txt
python data_gen.py
python train.py --epochs 40 --threads 2
python hybrid.py
python -m pytest tests serving/tests -q
python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000
```

Example API request: POST `/screen` with
`{"job_id": 0, "k": 10, "mode": "hybrid"}`. Other modes: `bm25`, `neural`.
The default remains `neural` for backward compatibility. Raw `job_text` is also
accepted instead of `job_id`. The API indexes the bundled corpus; use the local
import CLI below to score a different approved candidate corpus.

## Models and evaluation

- **BM25:** lexical baseline over the full candidate corpus.
- **Bi-encoder:** token embeddings, attention pooling, separate JD/resume towers,
  ListNet loss and an auxiliary skill-prediction loss. This is a small model trained
  from scratch, not a pretrained transformer or an LLM.
- **Hybrid:** weighted reciprocal-rank fusion avoids adding incompatible raw score
  scales. Choose among weights 0, 0.25, 0.5, 0.75, 1 using validation NDCG@10.
  Weight 0 is allowed: a more complex model is not required to win.

The 1,800 generated jobs are divided into 1,152 training, 288 validation and 360
existing test queries. The 3,000 synthetic resumes remain a fixed retrieval corpus.
Only training JDs plus candidate texts build the vocabulary; validation queries
select fusion weight; test queries report the final comparison. Checkpoints record
actual epochs, seed and batch size, and fusion configuration is fingerprinted to
its model/dataset artifacts.

**Measured results:** see [`reports/hybrid_results.json`](reports/hybrid_results.json).
This report includes validation selection, test metrics, environment and artifact
fingerprint. Older figures in `serving/` are historical smoke-test records and do
not describe the retrained model. See the separate local serving measurements below.

Limitations: template overlap across job splits, synthetic skill-overlap labels,
a previously inspected test set, one random seed and no real reviewer judgments.
The model truncates at 128 tokens and has a synthetic-trained vocabulary, so long
or unfamiliar real resumes require a different representation and evaluation.
Do not infer ATS superiority from these numbers.

## Score a local approved corpus

Input JSON schemas:

```json
[{"job_id": "job-001", "text": "Job requirements in plain text"}]
```

```json
[{"candidate_id": "candidate-001", "text": "De-identified professional experience"}]
```

```bash
python rank_documents.py --jobs /approved/local/jobs.json --candidates /approved/local/candidates.json --mode bm25 --system bm25 --out /approved/local/bm25.csv
python rank_documents.py --jobs /approved/local/jobs.json --candidates /approved/local/candidates.json --mode hybrid --neural-weight 0.5 --system prototype --out /approved/local/prototype.csv
```

The weight above is illustrative; choose it on validation jobs before final testing.
The CLI exports every candidate's score, not just the top results, and records input
hashes. It accepts pre-extracted text, not PDF/DOCX files. Keep company documents and
exports outside the public repository. No network calls, ATS writes or automatic
rejections occur. These scores are not calibrated probabilities of job suitability.

## New standalone evaluation project

[`ats_benchmark/`](ats_benchmark/README.md) compares a prototype and an actual ATS
export using blinded, adjudicated human relevance grades. It checks matching job
and candidate pools, rejects incomplete judgments and computes per-job metrics and
paired bootstrap confidence intervals. Its included example uses RANDOM synthetic
scores only to exercise the evaluator, not to simulate beating an ATS.

## Learning and interview preparation

- [Step-by-step learning path and interview questions (Chinese)](docs/LEARNING_PATH_ZH.md)
- [Engineering log: changes, verification and unresolved work](docs/ENGINEERING_LOG.md)

Use the project to learn and explain real implementation decisions. Describe it as
an AI-assisted personal prototype until company work and outcomes actually occur.

## Serving measurement

```bash
python serving/load_test.py --requests 200 --warmup 20 --concurrency 1 8 20 --mode hybrid
```

[`reports/load_results.json`](reports/load_results.json) records separate latency,
throughput and error counts for each concurrency level. It reuses HTTP connections,
excludes warmup and uses a closed-loop client on the same machine. This is a local
benchmark, not a production capacity or a directly comparable speedup over the old
client-per-request smoke test.
