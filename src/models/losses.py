"""
src/models/losses.py
----------------------
in_batch_contrastive_loss : used to train the two-tower model. Every other
    item in the batch acts as a free negative for the current user — no
    explicit negative sampling needed at training time, which is why this
    scales so well in production two-tower systems.

bpr_loss : used to train the matrix-factorization baseline, which needs
    explicit (pos, neg) pairs from src/data/preprocess.py's NegativeSampler.
"""

import torch
import torch.nn.functional as F


def in_batch_contrastive_loss(user_emb, item_emb, temperature=0.1, log_q=None):
    """log_q: (B,) log sampling prob of each batch item. Corrects u->i logits only."""
    batch_size = user_emb.size(0)
    logits = (user_emb @ item_emb.T) / temperature
    labels = torch.arange(batch_size, device=logits.device)
    u2i = logits - log_q[None, :] if log_q is not None else logits
    loss_u2i = F.cross_entropy(u2i, labels)
    loss_i2u = F.cross_entropy(logits.T, labels)
    return (loss_u2i + loss_i2u) / 2.0


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss: push score(pos) above score(neg).
    pos_scores, neg_scores: (batch_size,)
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()