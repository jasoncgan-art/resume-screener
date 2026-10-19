"""Train the bi-encoder retrieval model.

Usage:
    python train.py                      # full train -> artifacts/biencoder.pt
    python train.py --epochs 2           # quick smoke run

Training supervision (disclosed honestly -- this is weak supervision, the
kind you'd actually have):
    * For each train JD, a stratified list of 32 same-family resumes with
      graded relevance = required-skill overlap counts (8 high-overlap,
      8 medium, 8 low, 8 zero; backfilled if a bucket is short).
    * ListNet listwise loss: the model's softmax over predicted scores must
      match the softmax over the graded overlap relevance -- i.e. resumes
      covering more of the req's skills must score higher.
    * Plus an auxiliary multi-label skill-prediction loss on each tower
      (predict the text's canonical skills from its embedding), which forces
      the embeddings to compositionally encode *which* skills a text
      mentions. Skill heads are discarded at serving time.

Loss: ListNet + 2.0 * auxiliary skill BCE (pos_weight=20 for the rare
positive skill labels).

Pipeline:
    dataset.build_dataset()  (tokenized corpus + skill sets; saved once)
    -> build_train_examples() (JD, positive, hard negatives)
    -> train() -> artifacts/biencoder.pt {state_dict, config}

The BM25 baseline needs no training; evaluate.py builds its index from data/
directly, so both models are scored on the identical held-out JD split.

Determinism: seeds fixed for random/numpy/torch; DataLoader shuffling uses
a seeded torch.Generator. Re-running on the same machine gives the same
numbers.
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from dataset import build_dataset, pad_batch, save_dataset
from models import BiEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(BASE_DIR, "artifacts")

SEED = 42
DEVICE = torch.device("cpu")
N_HARD_NEGS = 3
RELEVANCE_OVERLAP = 4  # must match evaluate.py's MIN_SKILL_OVERLAP

CONFIG = {
    "emb_dim": 128,
    "hidden_dim": 256,
    "out_dim": 128,
    "temperature": 0.1,
    "batch_size": 32,
    "epochs": 40,
    "lr": 3e-3,
    "aux_weight": 2.0,
}


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def multihot(idxs_list: list[list[int]], n: int,
               device: torch.device) -> torch.Tensor:
    t = torch.zeros((len(idxs_list), n), dtype=torch.float32)
    for i, idxs in enumerate(idxs_list):
        t[i, idxs] = 1.0
    return t.to(device)


def build_rank_lists(data: dict, seed: int,
                   list_size: int = 32) -> list[tuple[int, list[int], list[int]]]:
    """Listwise training lists: for each train JD, a stratified sample of
    resumes with graded relevance = required-skill overlap counts.

    Stratification (per JD): 8 high-overlap (>=6), 8 medium (4-5),
    8 low (1-3), 8 zero-overlap, all same-family -- plus 8 cross-family
    resumes graded 0 (wrong family is never relevant). The model learns
    the full graded ranking via a ListNet loss -- directly the skill-coverage
    ordering the eval rewards -- instead of a single-positive contrastive
    task. The cross-family negatives teach the model the eval's actual
    candidate distribution (ALL resumes, not just same-family ones).
    """
    rng = random.Random(seed)
    jobs, resumes = data["jobs"], data["resumes"]
    rskillsets = [set(r["skills_expanded"]) for r in resumes]
    resume_family = data["resume_family"]
    n_res = len(resumes)
    lists = []
    for jid in data["train_job_ids"]:
        job = jobs[jid]
        req = set(job["req_skills"])
        buckets: dict[str, list[tuple[int, int]]] = {
            "high": [], "med": [], "low": [], "zero": []}
        for i, rs in enumerate(rskillsets):
            if resume_family[i] != job["family"]:
                continue
            ov = len(req & rs)
            key = ("high" if ov >= 6 else "med" if ov >= 4
                   else "low" if ov >= 1 else "zero")
            buckets[key].append((i, ov))
        n_fam = sum(1 for f in resume_family if f == job["family"])
        size = min(list_size, n_fam)
        per = max(size // 4, 1)
        chosen: list[tuple[int, int]] = []
        for key in ("high", "med", "low", "zero"):
            pool = [c for c in buckets[key] if c not in chosen]
            take = min(per, len(pool))
            chosen.extend(rng.sample(pool, take) if take else [])
        # backfill short buckets from the full same-family pool (duplicates
        # allowed only if the family is smaller than `size`)
        guard = 0
        while len(chosen) < size and guard < 10000:
            guard += 1
            i = rng.randrange(n_res)
            if resume_family[i] != job["family"]:
                continue
            ov = len(req & rskillsets[i])
            if (i, ov) not in chosen or len(chosen) >= n_fam:
                chosen.append((i, ov))
        rng.shuffle(chosen)
        # Cross-family negatives: the eval ranks ALL resumes, so the model
        # must learn that a wrong-family resume is never relevant, no matter
        # how many skills it shares. Without these, training only ever shows
        # same-family candidates and the model never learns to suppress
        # cross-family outliers (which then dominate top-1 at serving time).
        # Graded ov=0: family mismatch is never relevant per the eval
        # definition (same family AND >= 4 shared skills).
        n_cross = 8
        cross_pool = [i for i in range(n_res)
                      if resume_family[i] != job["family"]
                      and all(i != c[0] for c in chosen)]
        for i in rng.sample(cross_pool, min(n_cross, len(cross_pool))):
            chosen.append((i, 0))
        rng.shuffle(chosen)
        ridx = [c[0] for c in chosen]
        ovs = [c[1] for c in chosen]
        lists.append((jid, ridx, ovs))
    return lists


def train(data: dict, epochs: int = CONFIG["epochs"],
          batch_size: int = CONFIG["batch_size"], seed: int = SEED,
          verbose: bool = True, aux_weight: float | None = None,
          path: str | None = None) -> BiEncoder:
    """Train the bi-encoder. If `path` is given, save the checkpoint there;
    otherwise (e.g. smoke tests) the trained model is just returned."""
    if aux_weight is None:
        aux_weight = CONFIG["aux_weight"]
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(seed)
    rank_lists = build_rank_lists(data, seed)
    jobs, resumes = data["jobs"], data["resumes"]
    job_tokens, resume_tokens = data["job_tokens"], data["resume_tokens"]

    loader = DataLoader(
        TensorDataset(torch.arange(len(rank_lists))), batch_size=batch_size,
        shuffle=True, generator=torch.Generator().manual_seed(seed),
    )

    model = BiEncoder(data["vocab_size"], CONFIG["emb_dim"],
                      CONFIG["hidden_dim"], CONFIG["out_dim"],
                      CONFIG["temperature"],
                      n_skills=data["n_skills"]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])

    for epoch in range(epochs):
        model.train()
        total, n = 0.0, 0
        for (idx,) in loader:
            batch = [rank_lists[i] for i in idx.tolist()]
            jids = [b[0] for b in batch]
            # flatten: JD b pairs with resumes b*S .. b*S+S-1
            S = len(batch[0][1])
            r_flat = [r for b in batch for r in b[1]]
            ovs = torch.tensor([o for b in batch for o in b[2]],
                               dtype=torch.float32, device=DEVICE)
            jt = pad_batch([job_tokens[jid].tolist() for jid in jids])
            rt = pad_batch([resume_tokens[i].tolist() for i in r_flat])
            u, v, uj, vj = model(jt.to(DEVICE), rt.to(DEVICE),
                                 predict_skills=True)
            # ListNet: match the softmax over predicted scores to the
            # softmax over graded overlap relevance (textbook ListNet: the
            # target is softmax of the raw relevance grades, so strongly
            # relevant resumes get most of the probability mass and the
            # model must separate scores to match it).
            scores = (u.unsqueeze(1) * v.view(len(batch), S, -1)).sum(-1)
            scores = scores / model.temperature                      # (B, S)
            target = torch.softmax(ovs.view(len(batch), S), dim=1)
            listnet = -(target * torch.log_softmax(scores, dim=1)).sum(1).mean()
            # auxiliary skill prediction (JD + listed resumes)
            pw = torch.full((data["n_skills"],), 20.0, device=DEVICE)
            aux = (F.binary_cross_entropy_with_logits(
                       uj, multihot([data["job_skill_targets"][jid]
                                     for jid in jids],
                                    data["n_skills"], DEVICE), pos_weight=pw)
                   + F.binary_cross_entropy_with_logits(
                       vj, multihot([data["resume_skill_targets"][i]
                                     for i in r_flat],
                                    data["n_skills"], DEVICE), pos_weight=pw))
            loss = listnet + aux_weight * aux
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(batch)
            n += len(batch)
        if verbose:
            print(f"[train] epoch {epoch+1}/{epochs} loss={total/n:.4f}",
                  flush=True)

    if path is not None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({"state_dict": model.state_dict(),
                    "n_skills": data["n_skills"], **CONFIG,
                    "epochs": epochs, "batch_size": batch_size,
                    "seed": seed, "aux_weight": aux_weight}, path)
        print(f"[train] saved {path}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=CONFIG["epochs"])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    data = build_dataset()
    save_dataset(data)
    train(data, epochs=args.epochs,
          path=os.path.join(ART_DIR, "biencoder.pt"))


if __name__ == "__main__":
    main()

