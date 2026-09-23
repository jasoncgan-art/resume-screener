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

import torch

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

from dataset import encode, pad_batch  # noqa: E402
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
        try:
            job = self.data["jobs"][job_id]
        except IndexError:
            raise KeyError(job_id)
        return self.job_vector(job["text"])

    def screen(self, qvec: torch.Tensor, k: int) -> list[tuple[int, int, float]]:
        """Top-k over the resume index. Returns [(rank, resume_idx, score)]."""
        with torch.no_grad():
            scores = (self.resume_vectors @ qvec).double() - self._tie_break
            top = torch.topk(scores, k)
        return [(r + 1, int(i), float(s))
                for r, (i, s) in enumerate(zip(top.indices.tolist(),
                                               top.values.tolist()))]


def load_screener(settings: Settings | None = None) -> Screener:
    return Screener(settings)
