"""Training and evaluation loops for :class:`ssvae.SSVAE`.

``ssvae.py`` intentionally contains only the model; this module provides the
semi-supervised optimization used by the pipeline.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .experiment_config import SSVAEHyperparams
from .ssvae import SSVAE


def _cycle_batches(n: int, batch_size: int, device):
    """Yield index batches over ``n`` items forever, reshuffling after each pass."""
    while True:
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            yield perm[start:start + batch_size]


def fit_ssvae(
    model: SSVAE,
    x_labeled: torch.Tensor,
    y_labeled: torch.Tensor,
    x_unlabeled: torch.Tensor,
    hp: SSVAEHyperparams,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """Optimize the M2 objective with Adam.

    One epoch is a full pass over the unlabeled data (or over the labeled data
    if there is none).  Each step pairs one unlabeled batch with one labeled
    batch (the labeled set is cycled) and minimizes
    ``labeled_loss + unlabeled_loss``, both of which are per-item batch means.
    """
    if seed is not None:
        torch.manual_seed(seed)
    device = x_labeled.device
    n_l, n_u = x_labeled.shape[0], x_unlabeled.shape[0]
    if n_l == 0:
        raise ValueError("SSVAE training needs at least one labeled point.")

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    labeled_batches = _cycle_batches(n_l, hp.labeled_batch_size, device)
    history: dict[str, list[float]] = {"loss": [], "labeled_loss": [], "unlabeled_loss": [], "classification": []}

    for epoch in range(hp.epochs):
        model.train()
        if n_u > 0:
            u_perm = torch.randperm(n_u, device=device)
            n_steps = math.ceil(n_u / hp.unlabeled_batch_size)
        else:
            n_steps = math.ceil(n_l / hp.labeled_batch_size)

        sums = {k: 0.0 for k in history}
        for step in range(n_steps):
            l_idx = next(labeled_batches)
            labeled = model.labeled_loss(x_labeled[l_idx], y_labeled[l_idx], hp.classification_weight)
            loss = labeled["loss"]
            unlabeled_value = 0.0
            if n_u > 0:
                u_idx = u_perm[step * hp.unlabeled_batch_size:(step + 1) * hp.unlabeled_batch_size]
                unlabeled = model.unlabeled_loss(x_unlabeled[u_idx])
                loss = loss + unlabeled["loss"]
                unlabeled_value = unlabeled["loss"].item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            sums["loss"] += loss.item()
            sums["labeled_loss"] += labeled["loss"].item()
            sums["unlabeled_loss"] += unlabeled_value
            sums["classification"] += labeled["classification"].item()

        for k in history:
            history[k].append(sums[k] / n_steps)

        if verbose and (epoch % max(1, hp.epochs // 10) == 0 or epoch == hp.epochs - 1):
            print(
                f"[Epoch {epoch + 1}/{hp.epochs}] loss: {history['loss'][-1]:.4f}, "
                f"labeled: {history['labeled_loss'][-1]:.4f}, "
                f"unlabeled: {history['unlabeled_loss'][-1]:.4f}, "
                f"CE: {history['classification'][-1]:.4f}"
            )
    return history


@torch.no_grad()
def evaluate_ssvae(model: SSVAE, x: torch.Tensor, y: torch.Tensor, batch_size: int = 4096) -> dict[str, float]:
    """Classifier accuracy and NLL, -mean log q(y|x), on (x, y)."""
    model.eval()
    log_probs = torch.cat(
        [F.log_softmax(model.classify(x[i:i + batch_size]), dim=-1) for i in range(0, len(x), batch_size)]
    )
    return {
        "accuracy": (log_probs.argmax(dim=-1) == y).float().mean().item(),
        "nll": F.nll_loss(log_probs, y).item(),
    }
