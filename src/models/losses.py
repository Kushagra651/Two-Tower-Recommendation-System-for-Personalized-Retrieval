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


def in_batch_contrastive_loss(user_emb: torch.Tensor, item_emb: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    user_emb, item_emb: (batch_size, embedding_dim), L2-normalized.
    Row i of user_emb and row i of item_emb form the positive pair; every
    other row j != i is treated as a negative for user i (in-batch negatives).

    Symmetric loss (user->item and item->user) is more stable than one-directional.
    """
    batch_size = user_emb.size(0)
    logits = (user_emb @ item_emb.T) / temperature          # (batch_size, batch_size)
    labels = torch.arange(batch_size, device=logits.device)  # diagonal = positives

    loss_u2i = F.cross_entropy(logits, labels)
    loss_i2u = F.cross_entropy(logits.T, labels)
    return (loss_u2i + loss_i2u) / 2.0


def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    """
    Bayesian Personalized Ranking loss: push score(pos) above score(neg).
    pos_scores, neg_scores: (batch_size,)
    """
    return -F.logsigmoid(pos_scores - neg_scores).mean()