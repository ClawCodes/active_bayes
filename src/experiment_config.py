"""Experiment configuration for the successor-model training pipeline.

An :class:`ExperimentConfig` fully specifies one run: which dataset, how the
labeled subset of the pool is chosen, which successor model, its
hyperparameters, and the seed.  ``run_id()`` hashes everything that affects the
trained network, so each distinct configuration maps to exactly one stored run.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

DatasetName = Literal[
    "heteroscedastic", "multimodal", "energy", "concrete", "two_moons", "fashion_mnist"
]
ModelName = Literal["mdn", "ssvae"]

REGRESSION_DATASETS = {"heteroscedastic", "multimodal", "energy", "concrete"}
CLASSIFICATION_DATASETS = {"two_moons", "fashion_mnist"}
MODEL_TASK = {"mdn": "regression", "ssvae": "classification"}


@dataclass
class LabelConfig:
    """How the labeled subset of the pool is chosen.

    all:     every pool point is labeled (upper-bound baseline).
    random:  uniform sampling. A single seeded permutation of the pool is drawn
             and its first ``n_labeled`` entries are used, so budgets are nested
             (the 50-point set is contained in the 100-point set) like an
             actively acquired dataset.
    indices: reserved for actively acquired datasets (not implemented yet).
    """

    strategy: Literal["all", "random", "indices"] = "all"
    n_labeled: int | None = None
    indices_path: str | None = None

    def __post_init__(self) -> None:
        if self.strategy == "random" and (self.n_labeled is None or self.n_labeled < 1):
            raise ValueError("labels.strategy='random' requires a positive labels.n_labeled.")
        if self.strategy == "indices" and not self.indices_path:
            raise ValueError("labels.strategy='indices' requires labels.indices_path.")


@dataclass
class DataConfig:
    name: DatasetName
    n_samples: int = 2000  # synthetic datasets only
    test_fraction: float = 0.2  # ignored for fashion_mnist (official split)
    moons_noise: float = 0.18
    data_dir: str = "data"  # download/cache location; not part of the run id
    labels: LabelConfig = field(default_factory=LabelConfig)

    @property
    def task(self) -> str:
        return "regression" if self.name in REGRESSION_DATASETS else "classification"


@dataclass
class MDNHyperparams:
    n_components: int = 5
    hidden_dims: tuple[int, ...] = (64, 64)
    activation: Literal["tanh", "relu", "silu"] = "tanh"
    min_sigma: float = 1e-3
    lr: float = 1e-3
    epochs: int = 2000
    batch_size: int | None = None  # None = full batch
    weight_decay: float = 0.0
    grad_clip: float | None = 5.0


@dataclass
class SSVAEHyperparams:
    latent_dim: int = 16
    hidden_dims: tuple[int, ...] = (256, 256)
    likelihood: Literal["auto", "gaussian", "bernoulli"] = "auto"
    lr: float = 1e-3
    epochs: int = 50
    labeled_batch_size: int = 64
    unlabeled_batch_size: int = 256
    classification_weight: float = 1.0  # alpha
    weight_decay: float = 0.0


@dataclass
class ExperimentConfig:
    data: DataConfig
    model: ModelName
    mdn: MDNHyperparams = field(default_factory=MDNHyperparams)
    ssvae: SSVAEHyperparams = field(default_factory=SSVAEHyperparams)
    seed: int = 0
    device: str = "auto"  # not part of the run id
    output_root: str = "runs"  # not part of the run id

    def __post_init__(self) -> None:
        expected = MODEL_TASK[self.model]
        if self.data.task != expected:
            raise ValueError(
                f"Model '{self.model}' is a {expected} successor, but dataset "
                f"'{self.data.name}' is a {self.data.task} task (see proposal Table 1)."
            )

    def active_hyperparams(self) -> MDNHyperparams | SSVAEHyperparams:
        return self.mdn if self.model == "mdn" else self.ssvae

    def identity(self) -> dict:
        """Everything that determines the trained network."""
        data = asdict(self.data)
        data.pop("data_dir")
        return {
            "data": data,
            "model": self.model,
            "hyperparams": asdict(self.active_hyperparams()),
            "seed": self.seed,
        }

    def run_id(self) -> str:
        canonical = json.dumps(self.identity(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        data = dict(d["data"])
        data["labels"] = LabelConfig(**data.get("labels", {}))
        mdn = dict(d.get("mdn", {}))
        ssvae = dict(d.get("ssvae", {}))
        for hp in (mdn, ssvae):
            if "hidden_dims" in hp:
                hp["hidden_dims"] = tuple(hp["hidden_dims"])
        rest = {k: v for k, v in d.items() if k not in {"data", "mdn", "ssvae"}}
        return cls(
            data=DataConfig(**data),
            mdn=MDNHyperparams(**mdn),
            ssvae=SSVAEHyperparams(**ssvae),
            **rest,
        )
