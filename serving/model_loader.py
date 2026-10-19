"""Load trained artifacts once at startup; serve from in-memory tensors.

The bi-encoder is loaded from `biencoder.pt`, resume embeddings are
precomputed for the whole candidate corpus once (encode_resumes over all
resumes), and every request then costs one job-tower forward pass plus one
query-vector x resume-matrix matmul followed by an exact top-k over the
3,000 resumes. Deterministic: fixed torch seed, single-threaded matmuls
(torch_threads=1 in config), and a stable resume-index tie-break.
"""

from __future__ import annotations

import os
import sys
import json
import numpy as np

import torch

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from dataset import encode, pad_batch, tokenize  # noqa: E402
from models import BiEncoder  # noqa: E402
from serving.config import Settings, load_settings  # noqa: E402


class Screener:
    """Thread-safe read-only screener: model + precomputed resume index."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or load_settings()
        torch.manual_seed(self.settings.seed)
        # Thread knobs are process-global: setting them twice in one process
        # raises RuntimeError, so tolerate that (e.g. tests loading a second
        # screener in-process after the app's lifespan already did).
        for _set in (lambda: torch.set_num_threads(self.settings.torch_threads),
                     lambda: torch.set_num_interop_threads(self.settings.torch_threads)):
            try:
                _set()
            except RuntimeError:
                pass

        model_path = self.settings.resolved_model_path
        dataset_path = os.path.join(self.settings.resolved_artifact_dir, "dataset.pt")
        for p, name in [(model_path, "biencoder.pt"), (dataset_path, "dataset.pt")]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Missing required artifact {name} at {p}. "
                    "Artifacts are NOT retrained here; run `python train.py` first."
                )
        self.data = torch.load(dataset_path, weights_only=False)
        art = torch.load(model_path, weights_only=True)

        model = BiEncoder(
            self.data["vocab_size"], art["emb_dim"], art["hidden_dim"],
            art["out_dim"], temperature=art["temperature"],
            n_skills=art.get("n_skills", 0),
        )
        model.load_state_dict(art["state_dict"])
        model.eval()
        self.model = model
        self.n_resumes = len(self.data["resumes"])

        with torch.no_grad():
            # (N, d): the precomputed in-memory resume index, L2-normalized rows.
            self.resume_vectors = model.encode_resumes(
                self.data["resume_tokens"]).contiguous()
            # tiny per-index tie-break so equal scores order deterministically
            self._tie_break = torch.arange(self.n_resumes, dtype=torch.float64) * 1e-9

        from evaluate import BM25Index
        self.bm25 = BM25Index([tokenize(r["text"]) for r in self.data["resumes"]])
        self.hybrid_config = None
        hybrid_path = os.path.join(self.settings.resolved_artifact_dir, "hybrid_config.json")
        if os.path.exists(hybrid_path):
            from hybrid import artifact_fingerprint
            with open(hybrid_path) as f:
                self.hybrid_config = json.load(f)
            if self.hybrid_config["artifact_sha256"] != artifact_fingerprint(
                    model_path, dataset_path):
                raise ValueError("hybrid config is stale; rerun python hybrid.py after training")
        self.titles = [r["title"] for r in self.data["resumes"]]
        self.families = self.data["resume_family"]

    def job_vector(self, job_text: str) -> torch.Tensor:
        toks = encode(job_text, self.data["vocab"])
        if not toks:
            raise ValueError("job description produced no tokens")
        with torch.no_grad():
            return self.model.encode_jobs(pad_batch([toks]))[0]

    def job_vector_by_id(self, job_id: int) -> torch.Tensor:
        """Encode a corpus JD by id. Raises KeyError for unknown ids."""
        if not isinstance(job_id, int) or not 0 <= job_id < len(self.data["jobs"]):
            raise KeyError(job_id)
        try:
            job = self.data["jobs"][job_id]
        except IndexError:
            raise KeyError(job_id)
        return self.job_vector(job["text"])

    def screen(self, qvec: torch.Tensor, k: int) -> list[tuple[int, int, float]]:
        """Top-k over the resume index. Returns [(rank, resume_idx, score)]."""
        with torch.no_grad():
            if not 1 <= k <= self.n_resumes:
                raise ValueError("k exceeds available candidates")
            scores = (self.resume_vectors @ qvec).double()
            indices = torch.argsort(scores, descending=True, stable=True)[:k]
            top = (scores[indices], indices)
        return [(r + 1, int(i), float(s))
                for r, (i, s) in enumerate(zip(top[1].tolist(), top[0].tolist()))]

    def screen_text(self, text: str, k: int, mode: str = "hybrid"):
        from hybrid import fuse_scores, stable_order
        if not 1 <= k <= self.n_resumes:
            raise ValueError("k exceeds available candidates")
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("job description produced no tokens")
        lexical = self.bm25.scores(tokens)
        if mode == "bm25":
            scores = lexical
        elif mode in ("hybrid", "neural"):
            with torch.no_grad():
                semantic = (self.resume_vectors @ self.job_vector(text)).numpy()
            if mode == "neural":
                scores = semantic
            else:
                if self.hybrid_config is None:
                    raise ValueError("run python hybrid.py to select fusion on validation data")
                scores = fuse_scores(lexical, semantic,
                                     self.hybrid_config["neural_weight"])
        else:
            raise ValueError("unknown retrieval mode")
        order = stable_order(scores)[:k]
        return [(rank + 1, int(i), float(scores[i])) for rank, i in enumerate(order)]


def load_screener(settings: Settings | None = None) -> Screener:
    return Screener(settings)
