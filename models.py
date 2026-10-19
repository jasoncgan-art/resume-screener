"""Bi-encoder (two-tower) model for candidate-job matching.

Architecture (Deep Averaging Network style -- fast on CPU, no transformers):
    text -> token-id embedding -> attention-weighted pool over non-pad tokens
         -> MLP(emb_dim -> 256 -> out_dim) -> L2-normalized vector

Two towers with SEPARATE token-embedding tables (a job description and a
resume use language differently, so we don't force them to share the
lexicon), but identical shapes:
    JobTower:    JD text     -> job vector     (d,)
    ResumeTower: resume text -> resume vector  (d,)
    score(jd, resume) = <job_vec, resume_vec> / temperature   (cosine sim)

Training (see train.py): LISTWISE ListNet loss over stratified resume lists
(high/medium/low/zero skill-overlap) per JD -- the model learns the full
graded skill-coverage ranking, not just "one positive vs the rest". Plus an
AUXILIARY multi-label skill-prediction loss on each tower (predict the
text's canonical skills from its vector; the heads are discarded at serving
time), which forces the embeddings to compositionally encode *which* skills
a text mentions instead of smearing pair-level supervision. Skill labels are
derived from the same text the model reads and are never used at inference.

What's handled, what's not (documented honestly):
  * No logQ correction: unlike MovieLens, resume "popularity" is uniform by
    construction (families are balanced), so the popularity bias that
    motivated logQ in recsys-tower is negligible.
  * Negatives are sampled, not hard-mined from the full corpus -- production
    systems mine hard negatives; skipped here for clarity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextTower(nn.Module):
    """One tower: token embedding -> learned attention pool -> MLP -> L2 norm.

    Attention pooling (instead of mean pooling) lets the tower upweight
    informative tokens (skill phrases) and downweight template filler --
    important because a resume's 13 skills would otherwise drown in ~100
    tokens of boilerplate under a flat mean.
    """

    def __init__(self, vocab_size: int, emb_dim: int = 128,
                 hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.01)
        with torch.no_grad():
            self.emb.weight[0].zero_()  # <pad> contributes nothing
        self.attn = nn.Linear(emb_dim, 1)  # token importance score
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # token_ids: (B, L); pad token is 0
        mask = (token_ids != 0)                                # (B, L) bool
        e = self.emb(token_ids)                               # (B, L, E)
        scores = self.attn(e).squeeze(-1)                     # (B, L)
        if token_ids.ndim != 2 or token_ids.shape[1] == 0:
            raise ValueError("token sequences must have positive width")
        if (~mask.any(dim=1)).any():
            raise ValueError("empty/all-padding text cannot be encoded")
        scores = scores.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(scores, dim=1).unsqueeze(-1)    # (B, L, 1)
        pooled = (e * alpha).sum(dim=1)                       # (B, E)
        return F.normalize(self.mlp(pooled), p=2, dim=-1)


class BiEncoder(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 128,
                 hidden_dim: int = 256, out_dim: int = 128,
                 temperature: float = 0.1, n_skills: int = 0):
        super().__init__()
        self.job_tower = TextTower(vocab_size, emb_dim, hidden_dim, out_dim)
        self.resume_tower = TextTower(vocab_size, emb_dim, hidden_dim, out_dim)
        self.temperature = temperature
        self.n_skills = n_skills
        # Auxiliary skill-prediction heads (training only): force the
        # embeddings to compositionally encode *which* skills a text
        # mentions, instead of letting pair-level supervision smear the
        # signal. Discarded at serving time -- inference uses raw text only.
        self.job_skill_head = nn.Linear(out_dim, n_skills) if n_skills else None
        self.resume_skill_head = nn.Linear(out_dim, n_skills) if n_skills else None

    def forward(self, job_ids: torch.Tensor, resume_ids: torch.Tensor,
                predict_skills: bool = False):
        u, v = self.job_tower(job_ids), self.resume_tower(resume_ids)
        if not predict_skills or self.n_skills == 0:
            return u, v
        return u, v, self.job_skill_head(u), self.resume_skill_head(v)

    def inbatch_softmax_loss(self, u, v, neg_mask=None):
        """u, v: (B, d) L2-normalized job / resume vectors.

        neg_mask: optional (B, B) bool -- True where column j is a known
        relevant resume for row b's JD (same family, not the positive).
        Masked logits are set to -inf.
        """
        logits = (u @ v.t()) / self.temperature
        if neg_mask is not None:
            logits = logits.masked_fill(neg_mask, float("-inf"))
        labels = torch.arange(u.size(0), device=u.device)
        return F.cross_entropy(logits, labels)

    @torch.no_grad()
    def encode_jobs(self, job_ids: torch.Tensor) -> torch.Tensor:
        return self.job_tower(job_ids)

    @torch.no_grad()
    def encode_resumes(self, resume_ids: torch.Tensor) -> torch.Tensor:
        return self.resume_tower(resume_ids)

