# Engineering log

## 2026-10-19: evaluation and prototype foundation

Context: an HR manager requested a prototype to investigate dissatisfaction with
an existing ATS screener. No ATS output or manager judgments were supplied to this
repository; no company deployment or performance improvement is established.

Changes:
- Reserve validation jobs from the original training set; keep the previous test
  IDs fixed for reproducibility. Historical inspection means this test is not a
  new untouched company holdout.
- Add validation-selected weighted reciprocal rank fusion (BM25 + bi-encoder).
- Make vocabulary frequency ties deterministic instead of dependent on hash iteration order.
- Store actual training settings and reject stale fusion configurations via an
  artifact fingerprint. Reject empty embeddings and use stable score sorting.
- Add local-text score export and a separate strict ATS comparison evaluator.

Verification commands and measured metrics are in README and `reports/`.
Synthetic model results and random evaluator fixtures are different experiments.
Neither establishes that this prototype beats any commercial ATS.

Open work:
- Assess text extraction, long documents, out-of-vocabulary skills and synonyms.
- Gather blinded, job-related relevance judgments with reviewer agreement checks.
- Evaluate pretrained semantic retrieval and cross-encoder reranking on validation
  jobs before choosing whether their added cost is justified.
- Extend the recorded short local load test into a sustained serving benchmark.
- Validate applicability, data handling and the human review workflow with the company.

Verification completed: 34 unit/API tests passed; all three local document scoring
modes exported six finite scores for a two-job, three-candidate synthetic fixture.
The local hybrid HTTP benchmark completed 200 measured requests at each of
concurrency 1, 8 and 20 with zero errors; see `reports/load_results.json`.
