"""Build train/validation/test query partitions and model tensors.

Resumes are a fixed candidate corpus; only training JD texts and corpus texts
build the vocabulary. Frequency ties use lexical order for process-independent
IDs. Validation selects fusion weight, while test labels are reserved for reporting.
Training uses ListNet lists in train.py; legacy train_pairs remain for compatibility.
"""

from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
ART_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
MAX_VOCAB = 8000
MIN_DF = 2
MAX_LEN = 128  # tokens; synthetic texts are short, truncation is rare

TOKEN_RE = re.compile(r"[a-z]+")


def tokenize(text: str) -> list[str]:
    """Unigrams + bigrams. Bigrams (e.g. "machine_learning") let the model
    treat multi-word skills as single units -- important because most skills
    in the taxonomy are phrases, and mean-pooling over split unigrams
    dilutes them."""
    toks = TOKEN_RE.findall(text.lower())
    return toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]


def build_vocab(texts: list[str]) -> dict[str, int]:
    df = Counter()
    for t in texts:
        df.update(set(tokenize(t)))
    kept = [tok for tok, c in sorted(df.items(), key=lambda pair: (-pair[1], pair[0]))
            if c >= MIN_DF][:MAX_VOCAB]
    vocab = {"<pad>": 0, "<unk>": 1}
    for tok in kept:
        vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(t, vocab["<unk>"]) for t in tokenize(text)][:MAX_LEN]


def pad_batch(seqs: list[list[int]]) -> torch.Tensor:
    if not seqs:
        return torch.zeros((0, MAX_LEN), dtype=torch.long)
    width = max(1, max(len(s) for s in seqs))
    out = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


def load_corpus() -> tuple[list[dict], list[dict], dict]:
    with open(os.path.join(DATA_DIR, "jobs.json"), encoding="utf-8") as f:
        jobs = json.load(f)
    with open(os.path.join(DATA_DIR, "resumes.json"), encoding="utf-8") as f:
        resumes = json.load(f)
    with open(os.path.join(DATA_DIR, "split.json"), encoding="utf-8") as f:
        split = json.load(f)
    return jobs, resumes, split


def build_dataset(seed: int = SEED) -> dict:
    rng = random.Random(seed)
    jobs, resumes, split = load_corpus()
    job_by_id = {j["job_id"]: j for j in jobs}

    # Preserve the original test IDs. Reserve 20% of the OLD training jobs
    # per family for validation; no model selection uses test relevance.
    validation_ids = split.get("validation_job_ids")
    train_ids = list(split["train_job_ids"])
    if validation_ids is None:
        families = defaultdict(list)
        for jid in train_ids:
            families[job_by_id[jid]["family"]].append(jid)
        validation_ids, train_ids = [], []
        split_rng = random.Random(seed + 1)
        for family in sorted(families):
            ids = sorted(families[family])
            split_rng.shuffle(ids)
            count = max(1, len(ids) // 5) if len(ids) > 1 else 0
            validation_ids.extend(ids[:count])
            train_ids.extend(ids[count:])
    train_ids, validation_ids = sorted(train_ids), sorted(validation_ids)
    partitions = [set(train_ids), set(validation_ids), set(split["test_job_ids"])]
    if any(partitions[i] & partitions[j] for i in range(3) for j in range(i)):
        raise ValueError("train/validation/test job IDs overlap")
    train_jobs = [job_by_id[i] for i in train_ids]
    test_jobs = [job_by_id[i] for i in split["test_job_ids"]]

    # vocab from TRAIN texts only (no test leakage)
    vocab = build_vocab([j["text"] for j in train_jobs]
                        + [r["text"] for r in resumes])

    fam_of_resume = [r["family"] for r in resumes]
    by_family: dict[str, list[int]] = {}
    for i, r in enumerate(resumes):
        by_family.setdefault(r["family"], []).append(i)

    # one positive resume per train JD (same family, seeded sample)
    train_pairs = []
    for j in train_jobs:
        pos = rng.choice(by_family[j["family"]])
        train_pairs.append((j["job_id"], pos))

    def enc_jobs(js):
        return pad_batch([encode(j["text"], vocab) for j in js])

    def enc_resumes(idxs):
        return pad_batch([encode(resumes[i]["text"], vocab) for i in idxs])

    # canonical skill vocabulary + multi-hot targets for the auxiliary loss
    # (train.py). These labels are derived from the same text the model sees;
    # they are never used at inference time.
    skill_list = sorted({x for j in train_jobs for x in j["req_skills"]}
                        | {x for r in resumes for x in r["skills"]})
    skill_to_idx = {s: i for i, s in enumerate(skill_list)}
    job_skill_targets = [[skill_to_idx[x] for x in j["req_skills"] if x in skill_to_idx]
                         for j in jobs]
    resume_skill_targets = [[skill_to_idx[x] for x in r["skills"]]
                            for r in resumes]

    data = {
        "vocab": vocab,
        "vocab_size": len(vocab),
        "jobs": jobs,               # full list, index == position
        "resumes": resumes,         # full list, index == position
        "train_job_ids": train_ids,
        "validation_job_ids": validation_ids,
        "test_job_ids": split["test_job_ids"],
        "train_pairs": train_pairs,  # [(job_id, resume_idx)]
        "resume_family": fam_of_resume,
        "job_family": [j["family"] for j in jobs],
        # pre-tokenized tensors (variable width; re-padded per batch)
        "job_tokens": enc_jobs(jobs),
        "resume_tokens": enc_resumes(list(range(len(resumes)))),
        "seed": seed,
        "skill_list": skill_list,
        "n_skills": len(skill_list),
        "job_skill_targets": job_skill_targets,
        "resume_skill_targets": resume_skill_targets,
    }
    return data


def save_dataset(data: dict, path: str | None = None) -> str:
    os.makedirs(ART_DIR, exist_ok=True)
    path = path or os.path.join(ART_DIR, "dataset.pt")
    torch.save(data, path)
    return path


def main() -> None:
    data = build_dataset()
    path = save_dataset(data)
    n_unk_test = sum(
        1 for jid in data["test_job_ids"]
        for t in tokenize(data["jobs"][jid]["text"])
        if t not in data["vocab"]
    )
    print(f"[dataset] vocab={data['vocab_size']} train_pairs={len(data['train_pairs'])} "
          f"test_oov_tokens={n_unk_test} -> {path}")


if __name__ == "__main__":
    main()

