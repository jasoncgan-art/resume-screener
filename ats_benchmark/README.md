# ATS Ranking Benchmark Lab

A standalone, model-agnostic evaluation project alongside the screener. Its job is
to test whether a prototype orders a shared candidate pool better than an existing
system according to independently collected, job-related relevance judgments.
It does not require PyTorch, access an ATS, or make hiring decisions.

## Run the self-contained example

From the repository root, with Python and NumPy installed:

```bash
python -m ats_benchmark.demo
python -m ats_benchmark.evaluate --judgments ats_benchmark/examples/judgments.csv --runs ats_benchmark/examples/runs.csv --baseline synthetic_baseline --data-kind synthetic --k 5 --out ats_benchmark/examples/demo_report.json
```

The example uses RANDOM system scores and artificial relevance grades. Neither
system is an actual ATS or trained model. This tests the evaluation plumbing only.

## Input contract

`judgments.csv`: `job_id,candidate_id,grade,split`

- Grade 0: no evidence for the agreed job requirements; 1: partial evidence;
  2: sufficient evidence for review; 3: strong documented match. Missing information
  is not proof of absence. Agree on role-specific anchors before labeling.
- Split is train, validation or test. A job belongs to exactly one split.
- Each candidate in the evaluated pool needs a grade. Resolve disagreements
  separately before providing one adjudicated grade per job/candidate pair.

`runs.csv`: `system,job_id,candidate_id,score`

- Higher scores rank first. If an ATS exports only ranks, use negative rank as score.
- Every system must score the SAME candidates for every evaluated job. Duplicate
  IDs, missing scores, unjudged candidates, nonfinite scores and split overlap fail.
- Candidate IDs are opaque local IDs. Do not put names, resumes or contact data in
  these files. Do not commit company exports, even if pseudonymized.

## Study protocol for the manager

1. Record the actual ATS version/configuration, pain points and shortlist budget.
   Separate parsing failures from retrieval failures, ranking failures and unclear
   requirements; each calls for a different fix.
2. Freeze role requirements. Group related/duplicate requisitions into the same
   partition; use earlier jobs for development and new jobs for final testing.
3. Compare both systems on identical documents and the same candidate pool.
   Request full rankings when available. If only top results are available, label
   a pooled sample plus random candidates, but do NOT call its recall full-corpus
   recall. This tool deliberately reports recall WITHIN THE JUDGED POOL.
4. Have reviewers grade candidates in randomized order, blinded to the source
   system, without displaying the model score. Obtain a second review on a subset,
   measure agreement and adjudicate conflicts. Do not train solely on prior hires.
5. Select models and fusion weights on validation jobs. Freeze before final test.
6. Report per-job NDCG, Precision@K, pool recall, paired job bootstrap intervals,
   wins/losses by job and concrete error examples. The bootstrap assumes independent
   jobs; do not treat duplicated templates as independent evidence.
7. Run a shadow pilot only after appropriate company authorization. Keep human
   review and a way to inspect evidence. No automatic rejection or ATS writes.

## What a claim can say

With synthetic inputs: "Built and tested a reproducible ranking evaluation harness."
With a properly collected company benchmark: describe the tested jobs, date,
pool, metric, uncertainty and reviewer protocol. Offline agreement with reviewers
does not establish better hires, reduced time-to-fill or production impact.

Real data inputs and reports should live in an approved company environment,
outside this public repository. The CLI runs locally and makes no network calls.
