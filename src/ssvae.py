"""Reusable semi-supervised variational autoencoder (SSVAE).

This module implements the M2 model from Kingma et al. (2014):

    p(y) p(z) p(x | y, z)
    q(y | x) q(z | x, y)

It deliberately contains no dataset loading or training loop.  A caller only
needs to provide flattened observations and, for labeled batches, integer class
indices.  That keeps the model usable across tabular data, embeddings, and
image data that have been flattened by a dataset-specific wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F


Likelihood = Literal["gaussian", "bernoulli"]


@dataclass(frozen=True)
class SSVAEConfig:
    """Architecture and observation-model choices for :class:`SSVAE`.

    ``input_dim`` is the number of scalar features after flattening one sample.
    Use ``bernoulli`` only when every input feature is scaled to the interval
    [0, 1]; use ``gaussian`` for standardized or otherwise continuous features.
    """

    input_dim: int
    num_classes: int
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 256)
    likelihood: Likelihood = "gaussian"


@dataclass
class Reconstruction:
    """Parameters of the decoder's observation distribution."""

    mean_or_logits: Tensor
    log_variance: Tensor | None = None


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    """Build a small, dataset-independent fully connected network."""
    layers: list[nn.Module] = []
    current_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend((nn.Linear(current_dim, hidden_dim), nn.SiLU()))
        current_dim = hidden_dim
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class SSVAE(nn.Module):
    """M2 semi-supervised VAE for fixed-size vector observations.

    Example construction (not executed here)::

        model = SSVAE(SSVAEConfig(input_dim=784, num_classes=10,
                                  likelihood="bernoulli"))

    During training, optimize ``labeled_loss(...)`` on labeled batches and
    ``unlabeled_loss(...)`` on unlabeled batches, usually summed or weighted by
    the relative batch sizes.  Labels must be ``torch.long`` values in
    ``[0, num_classes)``.
    """

    def __init__(self, config: SSVAEConfig) -> None:
        super().__init__()
        if config.input_dim < 1 or config.num_classes < 2 or config.latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive; num_classes must be at least 2.")
        if not config.hidden_dims or any(dim < 1 for dim in config.hidden_dims):
            raise ValueError("hidden_dims must contain one or more positive dimensions.")

        self.config = config
        self.classifier = _mlp(config.input_dim, config.hidden_dims, config.num_classes)
        self.encoder = _mlp(
            config.input_dim + config.num_classes,
            config.hidden_dims,
            2 * config.latent_dim,
        )
        decoder_output_dim = config.input_dim * (2 if config.likelihood == "gaussian" else 1)
        self.decoder = _mlp(
            config.latent_dim + config.num_classes,
            tuple(reversed(config.hidden_dims)),
            decoder_output_dim,
        )

    def _flatten(self, x: Tensor) -> Tensor:
        """Flatten all non-batch dimensions and validate the configured size."""
        if x.ndim < 2:
            raise ValueError("x must have a batch dimension and at least one feature dimension.")
        x = x.flatten(start_dim=1)
        if x.shape[1] != self.config.input_dim:
            raise ValueError(f"Expected {self.config.input_dim} features, received {x.shape[1]}.")
        return x

    def classify(self, x: Tensor) -> Tensor:
        """Return q(y|x) logits; apply softmax for class probabilities."""
        return self.classifier(self._flatten(x))

    def encode(self, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
        """Return mean and log-variance for q(z | x, y)."""
        x = self._flatten(x)
        if y.ndim != 2 or y.shape != (x.shape[0], self.config.num_classes):
            raise ValueError("y must be one-hot/probabilistic with shape (batch_size, num_classes).")
        parameters = self.encoder(torch.cat((x, y.to(dtype=x.dtype)), dim=-1))
        return parameters.chunk(2, dim=-1)

    @staticmethod
    def reparameterize(mean: Tensor, log_variance: Tensor) -> Tensor:
        """Sample a differentiable latent variable from a diagonal Gaussian."""
        return mean + torch.randn_like(mean) * torch.exp(0.5 * log_variance)

    def decode(self, z: Tensor, y: Tensor) -> Reconstruction:
        """Return decoder distribution parameters for p(x | z, y)."""
        parameters = self.decoder(torch.cat((z, y.to(dtype=z.dtype)), dim=-1))
        if self.config.likelihood == "bernoulli":
            return Reconstruction(mean_or_logits=parameters)
        mean, log_variance = parameters.chunk(2, dim=-1)
        return Reconstruction(mean_or_logits=mean, log_variance=log_variance.clamp(-12.0, 8.0))

    def forward(self, x: Tensor, y: Tensor) -> tuple[Reconstruction, Tensor, Tensor]:
        """Run q(z|x,y), sample z, and decode; primarily useful for inspection."""
        mean, log_variance = self.encode(x, y)
        return self.decode(self.reparameterize(mean, log_variance), y), mean, log_variance

    def _reconstruction_nll(self, x: Tensor, reconstruction: Reconstruction) -> Tensor:
        """Negative log likelihood per item, up to observation-independent constants."""
        x = self._flatten(x)
        if self.config.likelihood == "bernoulli":
            return F.binary_cross_entropy_with_logits(
                reconstruction.mean_or_logits, x, reduction="none"
            ).sum(dim=-1)
        assert reconstruction.log_variance is not None
        precision_weighted_error = (x - reconstruction.mean_or_logits).square() * torch.exp(
            -reconstruction.log_variance
        )
        return 0.5 * (precision_weighted_error + reconstruction.log_variance).sum(dim=-1)

    @staticmethod
    def _standard_normal_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
        """KL[q(z|x,y) || N(0,I)] per item."""
        return 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance).sum(dim=-1)

    def _negative_elbo_given_y(self, x: Tensor, y: Tensor) -> Tensor:
        mean, log_variance = self.encode(x, y)
        reconstruction = self.decode(self.reparameterize(mean, log_variance), y)
        return self._reconstruction_nll(x, reconstruction) + self._standard_normal_kl(mean, log_variance)

    def labeled_loss(self, x: Tensor, labels: Tensor, classification_weight: float = 1.0) -> dict[str, Tensor]:
        """Return the labeled objective: negative ELBO plus supervised CE.

        ``classification_weight`` is often called alpha in SSVAE literature.
        The returned ``loss`` is a scalar suitable for ``loss.backward()``.
        """
        x = self._flatten(x)
        labels = labels.reshape(-1).long()
        if labels.shape[0] != x.shape[0]:
            raise ValueError("labels must contain one class index per input item.")
        if labels.numel() and (labels.min() < 0 or labels.max() >= self.config.num_classes):
            raise ValueError("labels contain a class outside [0, num_classes).")
        y = F.one_hot(labels, num_classes=self.config.num_classes).to(dtype=x.dtype)
        negative_elbo = self._negative_elbo_given_y(x, y).mean()
        classification = F.cross_entropy(self.classify(x), labels)
        return {
            "loss": negative_elbo + classification_weight * classification,
            "negative_elbo": negative_elbo.detach(),
            "classification": classification.detach(),
        }

    def unlabeled_loss(self, x: Tensor) -> dict[str, Tensor]:
        """Return the exact class-enumerated unlabeled M2 objective.

        All possible labels are evaluated for every item.  This is robust for a
        modest class count (e.g. classification datasets); for thousands of
        classes, replace enumeration with a sampling/relaxation strategy.
        """
        x = self._flatten(x)
        batch_size = x.shape[0]
        class_probabilities = self.classify(x).softmax(dim=-1)

        all_y = torch.eye(self.config.num_classes, device=x.device, dtype=x.dtype)
        repeated_x = x[:, None, :].expand(batch_size, self.config.num_classes, self.config.input_dim)
        repeated_y = all_y[None, :, :].expand(batch_size, self.config.num_classes, self.config.num_classes)
        class_negative_elbos = self._negative_elbo_given_y(
            repeated_x.reshape(-1, self.config.input_dim),
            repeated_y.reshape(-1, self.config.num_classes),
        ).reshape(batch_size, self.config.num_classes)

        expected_negative_elbo = (class_probabilities * class_negative_elbos).sum(dim=-1)
        negative_entropy = (class_probabilities * class_probabilities.clamp_min(1e-8).log()).sum(dim=-1)
        loss = (expected_negative_elbo + negative_entropy).mean()
        return {
            "loss": loss,
            "expected_negative_elbo": expected_negative_elbo.mean().detach(),
            "negative_entropy": negative_entropy.mean().detach(),
        }

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Return classifier probabilities q(y|x) for evaluation or acquisition."""
        return self.classify(x).softmax(dim=-1)

