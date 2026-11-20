"""On-disk storage of trained runs: exactly one directory per configuration.

Layout: ``<output_root>/<dataset>/<model>/<run_id>/``
    config.json           full ExperimentConfig
    model.pt              {"build_kwargs": ..., "state_dict": ...}
    history.json          training curves
    labeled_indices.npy   pool indices the model was trained on
    metrics.json          held-out test metrics (written by the test step)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .experiment_config import ExperimentConfig
from .models import MODELS


def run_dir(cfg: ExperimentConfig) -> Path:
    return Path(cfg.output_root) / cfg.data.name / cfg.model / cfg.run_id()


def run_exists(cfg: ExperimentConfig) -> bool:
    return (run_dir(cfg) / "model.pt").exists()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def save_run(
    cfg: ExperimentConfig,
    model: nn.Module,
    build_kwargs: dict,
    history: dict,
    labeled_idx: np.ndarray,
) -> Path:
    directory = run_dir(cfg)
    directory.mkdir(parents=True, exist_ok=True)
    _write_json(directory / "config.json", cfg.to_dict())
    _write_json(directory / "history.json", history)
    np.save(directory / "labeled_indices.npy", labeled_idx)
    torch.save({"build_kwargs": build_kwargs, "state_dict": model.state_dict()}, directory / "model.pt")
    # A retrained model invalidates old test metrics.
    (directory / "metrics.json").unlink(missing_ok=True)
    return directory


def save_metrics(cfg: ExperimentConfig, metrics: dict) -> None:
    _write_json(run_dir(cfg) / "metrics.json", metrics)


def load_run(cfg: ExperimentConfig, device) -> tuple[nn.Module, dict, np.ndarray]:
    """Rebuild the stored model and return (model, history, labeled_indices)."""
    directory = run_dir(cfg)
    checkpoint = torch.load(directory / "model.pt", map_location=device, weights_only=True)
    model = MODELS[cfg.model].construct(checkpoint["build_kwargs"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    history = json.loads((directory / "history.json").read_text())
    labeled_idx = np.load(directory / "labeled_indices.npy")
    return model, history, labeled_idx
