"""Evaluate BM25 (keyword baseline) vs the bi-encoder on held-out JDs.

Protocol (identical for both models -- this is what makes the comparison
fair): for each of the 360 held-out test JDs, rank ALL 3,000 resumes and
measure where the relevant ones land.

Relevance is FINE-GRAINED, not just family-level: a resume is relevant to a
JD iff it shares the JD's role family AND covers at least 4 of the JD's
required skills, where coverage counts both EXPLICIT skills and IMPLIED ones
(e.g. "PyTorch" implies "machine learning" -- see IMPLICATIONS in
data_gen.py). Surface forms don't matter at the label level, so a resume
saying "ML" still matches a JD asking for "machine learning". The
implication edges model inferred competence: the exact phenomenon keyword
search cannot capture (it only sees literal tokens) and semantic models can
learn from co-occurrence. This mirrors what a recruiter actually wants: the
candidates whose skills best cover *this* req's requirements, not just
anyone in the field.

Metrics: Recall@10, Recall@50, NDCG@10, MRR, averaged over test JDs.

BM25: the classic IR keyword baseline and a fair stand-in for the keyword
matching inside a typical ATS -- implemented here from scratch with numpy
(k1=1.5, b=0.75), scored over the exact same tokenization the neural model
sees. No strawman: it gets the full resume texts and tuned-standard
hyperparameters.

The bi-encoder is loaded from artifacts/biencoder.pt (frozen -- nothing is
retrained here). Train/eval alignment, disclosed honestly: the model trains
on graded skill-overlap lists (weak supervision -- overlap counts computed
from synthetic skill labels, the kind of signal you'd have in practice)
and is evaluated on fine-grained skill-overlap relevance (>= 4 shared
required skills). Same signal family, coarser at train time.

Outputs: comparison table on stdout + artifacts/eval_results.json.
"""

from __future__ import annotations

import json
import math
import os
from collections import Counter

import numpy as np
import torch

from dataset import build_dataset, pad_batch, tokenize
from models import BiEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
K1, B = 1.5, 0.75
MIN_SKILL_OVERLAP = 4  # relevance threshold (see module docstring)


# --------------------------------------------------------------------------- #
# BM25 (numpy, from scratch)
# --------------------------------------------------------------------------- #
class BM25Index:
    def __init__(self, docs_tokens: list[list[str]]):
        self.n_docs = len(docs_tokens)
        self.doc_lens = np.array([len(d) for d in docs_tokens], dtype=np.float64)
        self.avgdl = self.doc_lens.mean()
        self.df = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, toks in enumerate(docs_tokens):
            for t, c in Counter(toks).items():
                self.df[t] += 1
                self.postings.setdefault(t, []).append((i, c))
        # idf with +1 smoothing so every query term contributes non-negatively
        self.idf = {t: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)
                    for t, df in self.df.items()}

    def scores(self, query_tokens: list[str]) -> np.ndarray:
        s = np.zeros(self.n_docs, dtype=np.float64)
        for t in set(query_tokens):
            if t not in self.postings:
                continue
            idf = self.idf[t]
            for doc_i, tf in self.postings[t]:
                dl = self.doc_lens[doc_i]
                denom = tf + K1 * (1 - B + B * dl / self.avgdl)
                s[doc_i] += idf * tf * (K1 + 1) / denom
        return s


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def rank_metrics(scores: np.ndarray, relevant: set[int],
                 ks=(10, 50)) -> dict:
    """scores: (N,) higher = better. Deterministic tie-break by doc index."""
    order = np.lexsort((np.arange(len(scores)), -scores))
    pos = {doc: rank for rank, doc in enumerate(order) if doc in relevant}
    out = {}
    n_rel = len(relevant)
    for k in ks:
        hits = sum(1 for r in pos.values() if r < k)
        out[f"recall@{k}"] = hits / n_rel if n_rel else 0.0
    # NDCG@10 (binary relevance)
    k = 10
    dcg = sum(1.0 / math.log2(r + 2) for r in pos.values() if r < k)
    ideal = sum(1.0 / math.log2(r + 2) for r in range(min(k, n_rel)))
    out["ndcg@10"] = dcg / ideal if ideal else 0.0
    # MRR
    out["mrr"] = 1.0 / (min(pos.values()) + 1) if pos else 0.0
    return out


def relevant_resumes(data: dict, jid: int) -> set[int]:
    """Fine-grained relevance for a JD: same role family AND at least
    MIN_SKILL_OVERLAP of the JD's required skills covered (explicit or
    implied). Shared by evaluate.py and serving/offline_eval.py so the
    served numbers reproduce the offline ones."""
    job = data["jobs"][jid]
    req = set(job["req_skills"])
    resume_family = data["resume_family"]
    return {i for i, r in enumerate(data["resumes"])
            if resume_family[i] == job["family"]
            and len(req & set(r["skills_expanded"])) >= MIN_SKILL_OVERLAP}


def evaluate(data: dict, model: BiEncoder) -> dict:
    resumes = data["resumes"]
    resume_tokens = [tokenize(r["text"]) for r in resumes]
    test_ids = data["test_job_ids"]

    print("[evaluate] building BM25 index over "
          f"{len(resumes)} resumes ...")
    bm25 = BM25Index(resume_tokens)

    print("[evaluate] encoding resumes with bi-encoder ...")
    torch.manual_seed(SEED)
    with torch.no_grad():
        rvecs = model.encode_resumes(data["resume_tokens"]).numpy()  # (N, d)

    agg: dict[str, dict[str, float]] = {"bm25": {}, "bi-encoder": {}}
    counts = {"bm25": 0, "bi-encoder": 0}
    for name in agg:
        agg[name] = {"recall@10": 0.0, "recall@50": 0.0,
                     "ndcg@10": 0.0, "mrr": 0.0}

    for jid in test_ids:
        relevant = relevant_resumes(data, jid)
        # BM25
        s_bm25 = bm25.scores(tokenize(data["jobs"][jid]["text"]))
        # bi-encoder: cosine sim (vectors are L2-normalized)
        with torch.no_grad():
            jt = pad_batch([data["job_tokens"][jid].tolist()])
            qv = model.encode_jobs(jt).numpy()[0]
        s_nn = rvecs @ qv
        for name, s in (("bm25", s_bm25), ("bi-encoder", s_nn)):
            m = rank_metrics(s, relevant)
            for k_, v in m.items():
                agg[name][k_] += v
            counts[name] += 1

    for name in agg:
        for k_ in agg[name]:
            agg[name][k_] /= counts[name]
    return agg


def main() -> dict:
    data = build_dataset()
    art_path = os.path.join(ART_DIR, "biencoder.pt")
    if not os.path.exists(art_path):
        raise FileNotFoundError(f"Missing {art_path}; run `python train.py` first.")
    art = torch.load(art_path, map_location="cpu", weights_only=True)
    model = BiEncoder(data["vocab_size"], art["emb_dim"], art["hidden_dim"],
                      art["out_dim"], art["temperature"],
                      n_skills=art.get("n_skills", 0))
    model.load_state_dict(art["state_dict"])
    model.eval()

    agg = evaluate(data, model)
    os.makedirs(ART_DIR, exist_ok=True)
    with open(os.path.join(ART_DIR, "eval_results.json"), "w") as f:
        json.dump(agg, f, indent=2)

    print()
    print("model         Recall@10  Recall@50    NDCG@10      MRR")
    print("-------------------------------------------------------")
    for name in ("bm25", "bi-encoder"):
        m = agg[name]
        print(f"{name:<13} {m['recall@10']:.4f}    {m['recall@50']:.4f}    "
              f"{m['ndcg@10']:.4f}    {m['mrr']:.4f}")
    return agg


if __name__ == "__main__":
    main()
