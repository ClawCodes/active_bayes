"""Successor-model training pipeline: train -> test -> plot.

Usage (from the repository root):
    uv run python -m src.pipeline --config configs/mdn_heteroscedastic.yaml
    uv run python -m src.pipeline --config configs/ssvae_two_moons.yaml --steps '[train,test]'
    uv run python -m src.pipeline --config configs/mdn_concrete.yaml --mdn.lr 1e-4 --seed 3

Every step works from a :class:`RunContext`, which holds either a model that was
just trained in memory or one loaded from its stored run directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from . import run_store
from .datasets import DatasetSplit, load_dataset, select_labeled
from .experiment_config import ExperimentConfig
from .models import MODELS

Step = Literal["train", "test", "plot"]
ALL_STEPS: tuple[Step, ...] = ("train", "test", "plot")


@dataclass
class RunContext:
    cfg: ExperimentConfig
    device: torch.device
    split: DatasetSplit | None = None
    labeled_idx: np.ndarray | None = None
    model: nn.Module | None = None
    history: dict | None = None
    metrics: dict | None = None
    trained_this_session: bool = False
    extras: dict = field(default_factory=dict)

    @property
    def run_dir(self) -> Path:
        return run_store.run_dir(self.cfg)

    def ensure_data(self) -> DatasetSplit:
        if self.split is None:
            self.split = load_dataset(self.cfg.data, self.cfg.seed)
        return self.split

    def ensure_model(self) -> nn.Module:
        """Use the in-memory model, or load the stored run for this config."""
        if self.model is None:
            if not run_store.run_exists(self.cfg):
                raise FileNotFoundError(
                    f"No trained model at {self.run_dir}; run the 'train' step first."
                )
            self.model, self.history, self.labeled_idx = run_store.load_run(self.cfg, self.device)
        return self.model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def step_train(ctx: RunContext, overwrite: bool = False) -> None:
    cfg = ctx.cfg
    if run_store.run_exists(cfg) and not overwrite:
        print(f"[train] Reusing existing run {ctx.run_dir} (pass --overwrite true to retrain).")
        ctx.ensure_model()
        return

    split = ctx.ensure_data()
    ctx.labeled_idx = select_labeled(split, cfg.data.labels, cfg.seed)
    print(
        f"[train] {cfg.model} on {cfg.data.name}: pool={len(split.x_pool)}, "
        f"labeled={len(ctx.labeled_idx)}, test={len(split.x_test)}, seed={cfg.seed}"
    )

    spec = MODELS[cfg.model]
    hp = cfg.active_hyperparams()
    torch.manual_seed(cfg.seed)  # deterministic weight init
    build_kwargs = spec.build_kwargs(hp, split, cfg.data.name)
    ctx.model = spec.construct(build_kwargs).to(ctx.device)
    ctx.history = spec.train(ctx.model, hp, split, ctx.labeled_idx, cfg.seed, ctx.device)
    ctx.trained_this_session = True

    directory = run_store.save_run(cfg, ctx.model, build_kwargs, ctx.history, ctx.labeled_idx)
    print(f"[train] Saved run to {directory}")


def step_test(ctx: RunContext) -> None:
    model = ctx.ensure_model()
    split = ctx.ensure_data()
    ctx.metrics = MODELS[ctx.cfg.model].evaluate(model, split, ctx.device)
    run_store.save_metrics(ctx.cfg, ctx.metrics)
    formatted = ", ".join(f"{k}={v:.4f}" for k, v in ctx.metrics.items())
    print(f"[test] Held-out test metrics: {formatted}")


def step_plot(ctx: RunContext) -> None:
    # Placeholder: plotting will use ctx.model (trained in memory or loaded from
    # disk via ctx.ensure_model()), ctx.history, ctx.metrics, and ctx.split.
    print("[plot] Plotting is not implemented yet; skipping.")


def run(
    cfg: ExperimentConfig,
    steps: Sequence[Step] = ALL_STEPS,
    overwrite: bool = False,
) -> RunContext:
    """Run the requested steps (always in train -> test -> plot order)."""
    ctx = RunContext(cfg=cfg, device=resolve_device(cfg.device))
    for step in ALL_STEPS:
        if step not in steps:
            continue
        if step == "train":
            step_train(ctx, overwrite=overwrite)
        elif step == "test":
            step_test(ctx)
        elif step == "plot":
            step_plot(ctx)
    return ctx


def build_parser():
    from jsonargparse import ActionConfigFile, ArgumentParser

    parser = ArgumentParser(description="Train and evaluate MDN / SSVAE successor models.")
    parser.add_argument("--config", action=ActionConfigFile, help="YAML experiment config.")
    parser.add_class_arguments(ExperimentConfig, nested_key=None, as_group=False)
    parser.add_argument("--steps", type=list[Step], default=list(ALL_STEPS))
    parser.add_argument("--overwrite", type=bool, default=False)
    return parser


def main(argv: Sequence[str] | None = None) -> RunContext:
    parser = build_parser()
    args = parser.parse_args(argv).as_dict()
    steps = args.pop("steps")
    overwrite = args.pop("overwrite")
    args.pop("config", None)
    args.pop("__path__", None)
    cfg = ExperimentConfig.from_dict(args)
    return run(cfg, steps=steps, overwrite=overwrite)


if __name__ == "__main__":
    main()
