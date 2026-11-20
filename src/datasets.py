"""Uniform dataset access for the training pipeline.

Wraps the loaders in ``render_all_datasets.py`` and returns every dataset as a
pool / held-out test split of float32 numpy arrays.  Labeled-subset selection
inside the pool (uniform sampling, or later active learning) is done by
:func:`select_labeled`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from .experiment_config import DataConfig, LabelConfig
from .render_all_datasets import (
    load_concrete,
    load_energy_efficiency,
    load_fashion_mnist,
    make_heteroscedastic_regression,
    make_multimodal_regression,
)


@dataclass
class DatasetSplit:
    name: str
    task: str  # "regression" or "classification"
    x_pool: np.ndarray  # (N_pool, D) float32
    y_pool: np.ndarray  # (N_pool,) float32 for regression, int64 for classification
    x_test: np.ndarray
    y_test: np.ndarray
    num_classes: int | None = None

    @property
    def input_dim(self) -> int:
        return self.x_pool.shape[1]


def _load_full(cfg: DataConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the complete dataset (before any split) as (x, y)."""
    rng = np.random.default_rng(seed)
    if cfg.name == "heteroscedastic":
        x, y, _, _ = make_heteroscedastic_regression(cfg.n_samples, rng)
        return x.reshape(-1, 1), y
    if cfg.name == "multimodal":
        x, y = make_multimodal_regression(cfg.n_samples, rng)
        return x.reshape(-1, 1), y
    if cfg.name == "two_moons":
        return make_moons(n_samples=cfg.n_samples, noise=cfg.moons_noise, random_state=seed)
    if cfg.name == "energy":
        x, y, _, _ = load_energy_efficiency()
        return x.to_numpy(), y.to_numpy()
    if cfg.name == "concrete":
        x, y, _, _ = load_concrete()
        return x.to_numpy(), y.to_numpy()
    raise ValueError(f"Unknown dataset: {cfg.name}")


def load_dataset(cfg: DataConfig, seed: int) -> DatasetSplit:
    """Load ``cfg.name`` and split it into an acquisition pool and a held-out test set.

    The test set is never available for label selection or training.
    """
    if cfg.name == "fashion_mnist":
        # Official split: 60k training images form the pool, 10k test images are held out.
        root = Path(cfg.data_dir) / "fashion_mnist"
        x_pool, y_pool = load_fashion_mnist(root, train=True)
        x_test, y_test = load_fashion_mnist(root, train=False)
        x_pool = x_pool.reshape(len(x_pool), -1) / 255.0
        x_test = x_test.reshape(len(x_test), -1) / 255.0
    else:
        x, y = _load_full(cfg, seed)
        x_pool, x_test, y_pool, y_test = train_test_split(
            x, y, test_size=cfg.test_fraction, random_state=seed
        )

    if cfg.name == "two_moons":
        # Standardize with pool statistics so the Gaussian-likelihood SSVAE is well scaled.
        mean, std = x_pool.mean(axis=0), x_pool.std(axis=0)
        x_pool = (x_pool - mean) / std
        x_test = (x_test - mean) / std

    if cfg.task == "classification":
        y_dtype, num_classes = np.int64, int(np.max(y_pool)) + 1
    else:
        y_dtype, num_classes = np.float32, None

    return DatasetSplit(
        name=cfg.name,
        task=cfg.task,
        x_pool=np.asarray(x_pool, dtype=np.float32),
        y_pool=np.asarray(y_pool, dtype=y_dtype),
        x_test=np.asarray(x_test, dtype=np.float32),
        y_test=np.asarray(y_test, dtype=y_dtype),
        num_classes=num_classes,
    )


def uniform_acquisition_order(n_pool: int, seed: int) -> np.ndarray:
    """A seeded random ordering of the pool; its prefixes are nested uniform samples."""
    # Separate stream from the one used to generate synthetic data with the same seed.
    return np.random.default_rng([seed, 1]).permutation(n_pool)


def select_labeled(split: DatasetSplit, labels: LabelConfig, seed: int) -> np.ndarray:
    """Return the indices (into the pool) of the points whose labels are revealed."""
    n_pool = len(split.x_pool)
    if labels.strategy == "all":
        return np.arange(n_pool)
    if labels.strategy == "random":
        if labels.n_labeled > n_pool:
            raise ValueError(f"n_labeled={labels.n_labeled} exceeds pool size {n_pool}.")
        return np.sort(uniform_acquisition_order(n_pool, seed)[: labels.n_labeled])
    if labels.strategy == "indices":
        raise NotImplementedError(
            "Loading actively acquired indices is not implemented yet; "
            "this is where the BAL acquisition step will plug in."
        )
    raise ValueError(f"Unknown label strategy: {labels.strategy}")
