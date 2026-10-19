"""Validation-selected BM25/neural reciprocal-rank fusion. Synthetic benchmark only."""
from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import torch

from dataset import pad_batch, tokenize
from evaluate import BM25Index, rank_metrics, relevant_resumes
from models import BiEncoder

ROOT = Path(__file__).resolve().parent


def stable_order(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite vector")
    return np.lexsort((np.arange(len(scores)), -scores))


def fuse_scores(lexical, semantic, neural_weight, rank_constant=60):
    """Weighted RRF: score scales do not need to match. Weight chosen on validation."""
    if not 0 <= neural_weight <= 1 or rank_constant <= 0:
        raise ValueError("invalid fusion parameters")
    if len(lexical) != len(semantic) or len(lexical) == 0:
        raise ValueError("candidate universes must match and be nonempty")
    def reciprocal(scores):
        order = stable_order(scores)
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        return 1 / (rank_constant + ranks)
    return (1 - neural_weight) * reciprocal(lexical) + neural_weight * reciprocal(semantic)


def artifact_fingerprint(*paths):
    h = hashlib.sha256()
    for path in paths:
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def score_queries(data, model, query_ids):
    """Labels are read only for evaluation; retrievers receive text alone."""
    index = BM25Index([tokenize(r["text"]) for r in data["resumes"]])
    model.eval()
    with torch.no_grad():
        vectors = torch.cat([model.encode_resumes(part) for part in
                             data["resume_tokens"].split(256)]).numpy()
    rows = []
    for jid in query_ids:
        with torch.no_grad():
            query = model.encode_jobs(pad_batch([data["job_tokens"][jid].tolist()])).numpy()[0]
        rows.append((jid, index.scores(tokenize(data["jobs"][jid]["text"])),
                     vectors @ query, relevant_resumes(data, jid)))
    return rows


def summarize(rows, weight):
    values = [rank_metrics(fuse_scores(bm, nn, weight), relevant)
              for _, bm, nn, relevant in rows]
    if not values:
        raise ValueError("evaluation needs at least one query")
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]}


def select_weight(validation_rows, weights=(0.0, 0.25, 0.5, 0.75, 1.0)):
    results = [{"neural_weight": w, **summarize(validation_rows, w)} for w in weights]
    # On exact ties prefer the simpler lexical-only endpoint.
    best = max(results, key=lambda r: (r["ndcg@10"], -r["neural_weight"]))
    return best["neural_weight"], results


def main():
    torch.set_num_threads(2)
    artifact_dir = ROOT / "artifacts"
    dataset_path, model_path = artifact_dir / "dataset.pt", artifact_dir / "biencoder.pt"
    # Load only locally generated, trusted artifacts; pickle is not an upload format.
    data = torch.load(dataset_path, weights_only=False)
    art = torch.load(model_path, weights_only=True)
    model = BiEncoder(data["vocab_size"], art["emb_dim"], art["hidden_dim"],
                      art["out_dim"], art["temperature"], n_skills=art["n_skills"])
    model.load_state_dict(art["state_dict"])
    partitions = [set(data[key]) for key in
                  ("train_job_ids", "validation_job_ids", "test_job_ids")]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError("overlapping splits")
    validation = score_queries(data, model, data["validation_job_ids"])
    weight, tuning = select_weight(validation)
    config = {"schema_version": 1, "neural_weight": weight, "rank_constant": 60,
              "selection_metric": "validation NDCG@10", "seed": art["seed"],
              "artifact_sha256": artifact_fingerprint(model_path, dataset_path)}
    (artifact_dir / "hybrid_config.json").write_text(json.dumps(config, indent=2) + "\n")
    # Only now access the held-out test labels. Do not retune after this report.
    test = score_queries(data, model, data["test_job_ids"])
    report = {"data_type": "synthetic", "ats_comparison": "not performed",
              "split_counts": dict(zip(("train", "validation", "test"), map(len, partitions))),
              "config": config, "training_epochs": art["epochs"],
              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                              "numpy": np.__version__, "threads": 2},
              "validation": tuning,
              "test": {"bm25": summarize(test, 0), "neural": summarize(test, 1),
                       "selected_fusion": summarize(test, weight)},
              "limitations": ["Template-generated queries share a fixed synthetic candidate corpus.",
                              "One seed; no evidence of generalization to real resumes or ATS superiority.",
                              "Existing test set has been inspected historically; use a fresh company holdout."]}
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "hybrid_results.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
