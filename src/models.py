"""Model registry giving MDN and SSVAE a uniform build / train / evaluate interface."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from torch import nn

from . import mdn as mdn_module
from .datasets import DatasetSplit
from .experiment_config import MDNHyperparams, SSVAEHyperparams
from .ssvae import SSVAE, SSVAEConfig
from .ssvae_training import evaluate_ssvae, fit_ssvae

ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU, "silu": nn.SiLU}


@dataclass(frozen=True)
class ModelSpec:
    # (hyperparams, split, dataset_name) -> JSON-serializable constructor kwargs
    build_kwargs: Callable[..., dict]
    # constructor kwargs -> fresh module
    construct: Callable[[dict], nn.Module]
    # (model, hyperparams, split, labeled_idx, seed, device) -> history dict
    train: Callable[..., dict]
    # (model, split, device) -> metrics dict
    evaluate: Callable[..., dict]


# ----------------------------------------------------------------------------- MDN


def _mdn_build_kwargs(hp: MDNHyperparams, split: DatasetSplit, dataset_name: str) -> dict:
    return {
        "in_dim": split.input_dim,
        "out_dim": 1,
        "n_components": hp.n_components,
        "hidden_dims": list(hp.hidden_dims),
        "activation": hp.activation,
        "min_sigma": hp.min_sigma,
    }


def _mdn_construct(kwargs: dict) -> nn.Module:
    kwargs = dict(kwargs)
    kwargs["activation"] = ACTIVATIONS[kwargs["activation"]]
    kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
    return mdn_module.MDN(**kwargs)


def _mdn_train(model, hp: MDNHyperparams, split: DatasetSplit, labeled_idx: np.ndarray, seed: int, device) -> dict:
    # Only the labeled points are used; no validation data, the test set stays untouched.
    return mdn_module.fit(
        model,
        split.x_pool[labeled_idx],
        split.y_pool[labeled_idx],
        epochs=hp.epochs,
        lr=hp.lr,
        batch_size=hp.batch_size,
        weight_decay=hp.weight_decay,
        grad_clip=hp.grad_clip,
        verbose=True,
        seed=seed,
    )


def _mdn_evaluate(model, split: DatasetSplit, device) -> dict:
    return {
        "nll": model.nll(split.x_test, split.y_test),
        "rmse": mdn_module.rmse(model, split.x_test, split.y_test),
    }


# ----------------------------------------------------------------------------- SSVAE


def _ssvae_build_kwargs(hp: SSVAEHyperparams, split: DatasetSplit, dataset_name: str) -> dict:
    likelihood = hp.likelihood
    if likelihood == "auto":
        likelihood = "bernoulli" if dataset_name == "fashion_mnist" else "gaussian"
    return asdict(
        SSVAEConfig(
            input_dim=split.input_dim,
            num_classes=split.num_classes,
            latent_dim=hp.latent_dim,
            hidden_dims=tuple(hp.hidden_dims),
            likelihood=likelihood,
        )
    )


def _ssvae_construct(kwargs: dict) -> nn.Module:
    kwargs = dict(kwargs)
    kwargs["hidden_dims"] = tuple(kwargs["hidden_dims"])
    return SSVAE(SSVAEConfig(**kwargs))


def _ssvae_train(model, hp: SSVAEHyperparams, split: DatasetSplit, labeled_idx: np.ndarray, seed: int, device) -> dict:
    # Labeled: the selected pool points. Unlabeled: the rest of the pool, labels discarded.
    unlabeled_mask = np.ones(len(split.x_pool), dtype=bool)
    unlabeled_mask[labeled_idx] = False
    x = torch.as_tensor(split.x_pool, device=device)
    y = torch.as_tensor(split.y_pool, device=device)
    labeled_t = torch.as_tensor(labeled_idx, device=device)
    return fit_ssvae(
        model,
        x[labeled_t],
        y[labeled_t],
        x[torch.as_tensor(unlabeled_mask, device=device)],
        hp,
        seed=seed,
    )


def _ssvae_evaluate(model, split: DatasetSplit, device) -> dict:
    return evaluate_ssvae(
        model,
        torch.as_tensor(split.x_test, device=device),
        torch.as_tensor(split.y_test, device=device),
    )


MODELS: dict[str, ModelSpec] = {
    "mdn": ModelSpec(_mdn_build_kwargs, _mdn_construct, _mdn_train, _mdn_evaluate),
    "ssvae": ModelSpec(_ssvae_build_kwargs, _ssvae_construct, _ssvae_train, _ssvae_evaluate),
}
