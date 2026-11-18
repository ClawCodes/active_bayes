"""
Mixture Density Network (Bishop, 1994) for conditional density estimation.

Models p(y | x) = sum_k pi_k(x) N(y; mu_k(x), diag(sigma_k(x)^2)) and is trained
by minimizing the negative log of the mixture density (computed with log-sum-exp).

Example:
    from mdn import MDN, fit, rmse

    model = MDN(in_dim=1, out_dim=1, n_components=3)
    history = fit(model, x_train, y_train, epochs=2000, x_val=x_test, y_val=y_test)
    test_nll = model.nll(x_test, y_test)
    test_rmse = rmse(model, x_test, y_test)
"""

from __future__ import annotations

import copy
import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_tensor(a, device=None) -> torch.Tensor:
    """Convert numpy/torch input to a float32 tensor of shape (N, D)."""
    t = torch.as_tensor(np.asarray(a) if not torch.is_tensor(a) else a, dtype=torch.float32)
    if t.ndim == 1:
        t = t.view(-1, 1)
    return t.to(device) if device is not None else t


def mdn_nll_loss(
    log_pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    y: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Negative log-likelihood of y under a diagonal Gaussian mixture.

    log_pi: (N, K) log mixture weights
    mu:     (N, K, D) component means
    sigma:  (N, K, D) component standard deviations
    y:      (N, D) targets
    reduction: "mean", "sum", or "none"
    """
    y = y.unsqueeze(1)  # (N, 1, D), broadcasts over components
    comp_log_prob = (
        -0.5 * ((y - mu) / sigma) ** 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)
    ).sum(-1)  # (N, K)
    nll = -torch.logsumexp(log_pi + comp_log_prob, dim=1)  # (N,)

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"Unsupported reduction: {reduction}")


class MDN(nn.Module):
    """
    MLP trunk with three heads producing mixture weights, means, and standard deviations.

    in_dim:       input dimension
    out_dim:      target dimension D
    n_components: number of mixture components K
    hidden_dims:  hidden layer widths of the shared trunk
    activation:   activation module class used in the trunk
    min_sigma:    lower bound on component standard deviations (normalized space)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 1,
        n_components: int = 5,
        hidden_dims: Sequence[int] = (64, 64),
        activation: type[nn.Module] = nn.Tanh,
        min_sigma: float = 1e-3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_components = n_components
        self.min_sigma = min_sigma

        layers = []
        prev = in_dim
        for width in hidden_dims:
            layers += [nn.Linear(prev, width), activation()]
            prev = width
        self.trunk = nn.Sequential(*layers)

        self.pi_head = nn.Linear(prev, n_components)
        self.mu_head = nn.Linear(prev, n_components * out_dim)
        self.sigma_head = nn.Linear(prev, n_components * out_dim)

        # Normalization statistics, set by fit(); identity by default
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))
        self.register_buffer("y_mean", torch.zeros(out_dim))
        self.register_buffer("y_std", torch.ones(out_dim))

    @property
    def device(self) -> torch.device:
        return self.x_mean.device

    def set_normalization(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Store per-feature mean/std of the training data."""
        def stats(t):
            mean = t.mean(dim=0)
            std = t.std(dim=0, unbiased=False) if t.shape[0] > 1 else torch.ones_like(mean)
            return mean, torch.where(std < 1e-8, torch.ones_like(std), std)

        self.x_mean, self.x_std = stats(x.to(self.device))
        self.y_mean, self.y_std = stats(y.to(self.device))

    def _normalize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def _normalize_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def forward(self, x: torch.Tensor):
        """
        Mixture parameters in normalized space for already-normalized x.

        Returns log_pi (N, K), mu (N, K, D), sigma (N, K, D).
        """
        h = self.trunk(x)
        log_pi = F.log_softmax(self.pi_head(h), dim=-1)
        mu = self.mu_head(h).view(-1, self.n_components, self.out_dim)
        sigma = F.softplus(self.sigma_head(h)).view(-1, self.n_components, self.out_dim) + self.min_sigma
        return log_pi, mu, sigma

    def log_prob(self, x, y) -> torch.Tensor:
        """Per-point log p(y | x) in original units, shape (N,)."""
        x = _to_tensor(x, self.device)
        y = _to_tensor(y, self.device)
        log_pi, mu, sigma = self(self._normalize_x(x))
        log_p_norm = -mdn_nll_loss(log_pi, mu, sigma, self._normalize_y(y), reduction="none")
        # Change of variables from normalized to original y
        return log_p_norm - torch.log(self.y_std).sum()

    @torch.no_grad()
    def nll(self, x, y) -> float:
        """Mean negative log-likelihood in original units."""
        self.eval()
        return -self.log_prob(x, y).mean().item()

    @torch.no_grad()
    def mixture_params(self, x):
        """Mixture weights (N, K), means (N, K, D), and stddevs (N, K, D) in original units."""
        self.eval()
        x = _to_tensor(x, self.device)
        log_pi, mu, sigma = self(self._normalize_x(x))
        return log_pi.exp(), mu * self.y_std + self.y_mean, sigma * self.y_std

    @torch.no_grad()
    def predict(self, x) -> torch.Tensor:
        """Point prediction: mean of the most heavily weighted component, shape (N, D)."""
        pi, mu, _ = self.mixture_params(x)
        best = pi.argmax(dim=1)
        return mu[torch.arange(mu.shape[0], device=mu.device), best]

    @torch.no_grad()
    def mixture_mean(self, x) -> torch.Tensor:
        """Mean of the full mixture, shape (N, D)."""
        pi, mu, _ = self.mixture_params(x)
        return (pi.unsqueeze(-1) * mu).sum(dim=1)

    @torch.no_grad()
    def mixture_variance(self, x) -> torch.Tensor:
        """Variance of the full mixture (law of total variance), shape (N, D)."""
        pi, mu, sigma = self.mixture_params(x)
        w = pi.unsqueeze(-1)
        mean = (w * mu).sum(dim=1)
        return (w * (sigma ** 2 + mu ** 2)).sum(dim=1) - mean ** 2

    @torch.no_grad()
    def sample(self, x, n_samples: int = 1) -> torch.Tensor:
        """Draw samples y ~ p(y | x), shape (n_samples, N, D)."""
        pi, mu, sigma = self.mixture_params(x)
        comp = torch.distributions.Categorical(probs=pi).sample((n_samples,))  # (S, N)
        idx = comp.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, self.out_dim)  # (S, N, 1, D)
        mu_s = mu.unsqueeze(0).expand(n_samples, -1, -1, -1).gather(2, idx).squeeze(2)
        sigma_s = sigma.unsqueeze(0).expand(n_samples, -1, -1, -1).gather(2, idx).squeeze(2)
        return mu_s + sigma_s * torch.randn_like(mu_s)


def fit(
    model: MDN,
    x,
    y,
    epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int | None = None,
    weight_decay: float = 0.0,
    x_val=None,
    y_val=None,
    standardize: bool = True,
    grad_clip: float | None = 5.0,
    verbose: bool = False,
    seed: int | None = None,
) -> dict:
    """
    Train an MDN by minimizing the mixture negative log-likelihood with Adam.

    batch_size=None trains full-batch. If validation data is given, the weights with
    the lowest validation NLL are restored at the end. Reported NLLs are in original units.

    Returns {"train_nll": [...], "val_nll": [...]}.
    """
    if seed is not None:
        torch.manual_seed(seed)

    device = model.device
    x = _to_tensor(x, device)
    y = _to_tensor(y, device)
    has_val = x_val is not None and y_val is not None

    if standardize:
        model.set_normalization(x, y)
    xn, yn = model._normalize_x(x), model._normalize_y(y)
    log_jacobian = torch.log(model.y_std).sum().item()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    n = x.shape[0]
    batch_size = n if batch_size is None else min(batch_size, n)

    history = {"train_nll": [], "val_nll": []}
    best_val, best_state = float("inf"), None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            loss = mdn_nll_loss(*model(xn[idx]), yn[idx])
            loss.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            epoch_loss += loss.item() * idx.numel()

        history["train_nll"].append(epoch_loss / n + log_jacobian)

        if has_val:
            val_nll = model.nll(x_val, y_val)
            history["val_nll"].append(val_nll)
            if val_nll < best_val:
                best_val = val_nll
                best_state = copy.deepcopy(model.state_dict())

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1):
            msg = f"[Epoch {epoch + 1}/{epochs}] train NLL: {history['train_nll'][-1]:.4f}"
            if has_val:
                msg += f", val NLL: {history['val_nll'][-1]:.4f}"
            print(msg)

    if best_state is not None:
        model.load_state_dict(best_state)

    return history


@torch.no_grad()
def rmse(model: MDN, x, y) -> float:
    """RMSE of the most-heavily-weighted-component mean against y."""
    y = _to_tensor(y, model.device)
    return torch.sqrt(torch.mean((model.predict(x) - y) ** 2)).item()
