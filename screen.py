"""Screen candidates for a job: rank resumes by bi-encoder cosine similarity.

Usage:
    python screen.py --job_id 7 --top_n 10
    python screen.py --job_text "Senior Data Scientist. Required skills: Python, statistics, A/B testing."
"""

from __future__ import annotations

import argparse
import os

import torch

from dataset import build_dataset, encode, pad_batch, tokenize
from models import BiEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")


def load_model_and_data():
    data = build_dataset()
    art = torch.load(os.path.join(ART_DIR, "biencoder.pt"),
                     map_location="cpu", weights_only=True)
    model = BiEncoder(data["vocab_size"], art["emb_dim"], art["hidden_dim"],
                      art["out_dim"], art["temperature"],
                      n_skills=art.get("n_skills", 0))
    model.load_state_dict(art["state_dict"])
    model.eval()
    with torch.no_grad():
        rvecs = model.encode_resumes(data["resume_tokens"])
    return data, model, rvecs


def screen(data, model, rvecs, job_text: str, top_n: int = 10):
    toks = encode(job_text, data["vocab"])
    with torch.no_grad():
        qv = model.encode_jobs(pad_batch([toks]))
        scores = (rvecs @ qv[0]).tolist()
    order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
    return [(i, scores[i]) for i in order[:top_n]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job_id", type=int, default=None,
                    help="screen for a corpus JD by id")
    ap.add_argument("--job_text", type=str, default=None,
                    help="screen for raw JD text (overrides --job_id)")
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()
    if args.job_text is None and args.job_id is None:
        ap.error("provide --job_id or --job_text")

    data, model, rvecs = load_model_and_data()
    if args.job_text is not None:
        job_text, label = args.job_text, "custom JD"
    else:
        job = data["jobs"][args.job_id]
        job_text, label = job["text"], f"job {args.job_id} ({job['title']})"

    print(f"[screen] {label}: {job_text[:120]}...")
    for rank, (i, s) in enumerate(screen(data, model, rvecs, job_text,
                                         args.top_n), start=1):
        r = data["resumes"][i]
        mark = ("*" if r["family"] == data["jobs"][args.job_id]["family"]
                else "") if args.job_text is None else ""
        print(f"  {rank:2d}. candidate-{r['candidate_id']:04d} "
              f"[{r['title']}] score={s:.4f} {mark}")
    if args.job_text is None:
        print("(* = same role family as the JD; relevance also needs >= 4 "
              "shared required skills)")


if __name__ == "__main__":
    main()
