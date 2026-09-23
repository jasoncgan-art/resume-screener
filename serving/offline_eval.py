"""Offline evaluation of the SERVED artifacts.

Runs the same held-out-JD protocol as evaluate.py, but through the serving
loader (Screener): encodes test JDs with the job tower and ranks the
precomputed resume index. Guards against train/serve skew -- the numbers
here should reproduce evaluate.py's bi-encoder row up to float noise.

Usage:  python serving/offline_eval.py [--out serving/eval_results.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

SERVING_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.normpath(os.path.join(SERVING_DIR, ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from serving.model_loader import load_screener  # noqa: E402
from evaluate import relevant_resumes  # noqa: E402 (shared relevance def)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(SERVING_DIR, "eval_results.json"))
    args = ap.parse_args()

    sc = load_screener()
    data = sc.data
    test_ids = data["test_job_ids"]

    agg = {"recall@10": 0.0, "recall@50": 0.0, "ndcg@10": 0.0, "mrr": 0.0}
    for jid in test_ids:
        relevant = relevant_resumes(data, jid)
        qvec = sc.job_vector_by_id(jid)
        res = sc.screen(qvec, len(data["resumes"]))
        pos = [rank - 1 for rank, ridx, _ in res if ridx in relevant]
        n_rel = len(relevant)
        for k in (10, 50):
            agg[f"recall@{k}"] += sum(1 for r in pos if r < k) / n_rel
        dcg = sum(1.0 / math.log2(r + 2) for r in pos if r < 10)
        ideal = sum(1.0 / math.log2(r + 2) for r in range(min(10, n_rel)))
        agg["ndcg@10"] += dcg / ideal if ideal else 0.0
        agg["mrr"] += 1.0 / (min(pos) + 1) if pos else 0.0
    n = len(test_ids)
    agg = {k: round(v / n, 4) for k, v in agg.items()}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"bi-encoder (served)": agg, "n_test_jds": n}, f, indent=2)
    print(json.dumps(agg, indent=2))
    print(f"[offline_eval] wrote {args.out}")


if __name__ == "__main__":
    main()
