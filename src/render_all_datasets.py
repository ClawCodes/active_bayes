#!/usr/bin/env python3
"""Render every synthetic and real-world benchmark used in the report.

Outputs (by default) are saved in figures/datasets/.  The script deliberately
uses every sample only for visualization; create the train/pool/test split in
the active-learning experiment, not here.

Install dependencies:
    pip install numpy pandas matplotlib scikit-learn ucimlrepo torch torchvision

Run:
    python render_all_datasets.py
    python render_all_datasets.py --output-dir figures/datasets --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons


FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


def set_plot_style() -> None:
    """Use a consistent, report-friendly Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 11,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def save_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


def make_heteroscedastic_regression(
    n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate y = sin(3x) + epsilon with input-dependent noise."""
    x = rng.uniform(-1.0, 1.0, size=n_samples)
    mean = np.sin(3.0 * x)
    noise_std = 0.05 + 0.30 * np.abs(x)
    y = mean + rng.normal(0.0, noise_std)
    return x, y, mean, noise_std


def render_heteroscedastic_regression(output_dir: Path, rng: np.random.Generator) -> None:
    x, y, _, _ = make_heteroscedastic_regression(450, rng)
    x_curve = np.linspace(-1.0, 1.0, 600)
    mean_curve = np.sin(3.0 * x_curve)
    std_curve = 0.05 + 0.30 * np.abs(x_curve)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(x, y, s=17, alpha=0.45, color="tab:blue", edgecolor="none", label="Samples")
    ax.plot(x_curve, mean_curve, color="black", lw=2.2, label=r"Mean $\sin(3x)$")
    ax.fill_between(
        x_curve,
        mean_curve - 2.0 * std_curve,
        mean_curve + 2.0 * std_curve,
        color="tab:orange",
        alpha=0.25,
        label=r"Noise band $\pm 2\sigma(x)$",
    )
    ax.set(
        title="Synthetic heteroscedastic regression",
        xlabel=r"Input $x$",
        ylabel=r"Target $y$",
    )
    ax.legend(loc="best")
    save_figure(fig, output_dir / "synthetic_heteroscedastic_regression.png")


def make_multimodal_regression(
    n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an equal-weight two-component conditional Gaussian mixture."""
    x = rng.uniform(-np.pi, np.pi, size=n_samples)
    component = rng.integers(0, 2, size=n_samples)
    component_mean = np.where(component == 0, np.sin(2.0 * x), -np.sin(2.0 * x))
    y = component_mean + rng.normal(0.0, 0.10, size=n_samples)
    return x, y


def render_multimodal_regression(output_dir: Path, rng: np.random.Generator) -> None:
    x, y = make_multimodal_regression(600, rng)
    x_curve = np.linspace(-np.pi, np.pi, 700)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.scatter(x, y, s=16, alpha=0.42, color="tab:purple", edgecolor="none", label="Samples")
    ax.plot(x_curve, np.sin(2.0 * x_curve), color="black", lw=2.0, label=r"Mode $+\sin(2x)$")
    ax.plot(x_curve, -np.sin(2.0 * x_curve), color="black", lw=2.0, ls="--", label=r"Mode $-\sin(2x)$")
    ax.set(
        title="Synthetic multimodal regression",
        xlabel=r"Input $x$",
        ylabel=r"Target $y$",
    )
    ax.legend(loc="best")
    save_figure(fig, output_dir / "synthetic_multimodal_regression.png")


def render_two_moons(output_dir: Path, seed: int) -> None:
    x, y = make_moons(n_samples=600, noise=0.18, random_state=seed)
    fig, ax = plt.subplots(figsize=(5.7, 4.5))
    points = ax.scatter(x[:, 0], x[:, 1], c=y, cmap="coolwarm", s=22, alpha=0.82, edgecolor="none")
    handles, _ = points.legend_elements()
    ax.legend(handles, ["Class 0", "Class 1"], title="Label", loc="best")
    ax.set(
        title="Two-moons classification dataset",
        xlabel=r"Feature $x_1$",
        ylabel=r"Feature $x_2$",
        aspect="equal",
    )
    save_figure(fig, output_dir / "two_moons.png")


def find_column(frame: pd.DataFrame, candidate_names: list[str]) -> str:
    """Find a UCI column despite minor differences in API column names."""
    def normalise(name: object) -> str:
        return re.sub(r"[^a-z0-9]", "", str(name).lower())

    normalized_columns = {normalise(column): str(column) for column in frame.columns}
    for candidate in candidate_names:
        candidate_key = normalise(candidate)
        if candidate_key in normalized_columns:
            return normalized_columns[candidate_key]
        for key, original in normalized_columns.items():
            if candidate_key in key or key in candidate_key:
                return original
    raise KeyError(
        f"Could not find any of {candidate_names}. Available columns: {list(frame.columns)}"
    )


def fetch_uci_dataset(dataset_id: int):
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError as exc:
        raise ImportError(
            "Install the UCI loader with `pip install ucimlrepo`."
        ) from exc
    return fetch_ucirepo(id=dataset_id)


def load_energy_efficiency() -> tuple[pd.DataFrame, pd.Series, str, str]:
    """Load UCI Energy Efficiency and select Heating Load (Y1) as the target."""
    dataset = fetch_uci_dataset(242)
    x = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    feature = find_column(x, ["Relative Compactness", "X1"])
    target = find_column(targets, ["Y1", "Heating Load"])
    return x, targets[target], feature, target


def load_concrete() -> tuple[pd.DataFrame, pd.Series, str, str]:
    """Load UCI Concrete Compressive Strength and select Cement as a feature."""
    dataset = fetch_uci_dataset(165)
    x = dataset.data.features.copy()
    targets = dataset.data.targets.copy()
    feature = find_column(x, ["Cement", "Cement (component 1)"])
    target = find_column(targets, ["Concrete compressive strength", "Strength"])
    return x, targets[target], feature, target


def render_tabular_regression(
    x: pd.DataFrame,
    y: pd.Series,
    feature: str,
    target: str,
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.3))
    ax.scatter(x[feature], y, s=22, alpha=0.55, color="tab:green", edgecolor="none")
    ax.set(title=title, xlabel=feature, ylabel=target)
    save_figure(fig, output_path)


def load_fashion_mnist(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Download/load Fashion-MNIST's training split with torchvision."""
    try:
        from torchvision.datasets import FashionMNIST
    except ImportError as exc:
        raise ImportError(
            "Install PyTorch and torchvision; see https://pytorch.org/get-started/locally/."
        ) from exc

    dataset = FashionMNIST(root=str(data_dir), train=True, download=True)
    return dataset.data.numpy(), dataset.targets.numpy()


def render_fashion_mnist(output_dir: Path, data_dir: Path, rng: np.random.Generator) -> None:
    images, labels = load_fashion_mnist(data_dir)
    selected = rng.choice(len(images), size=25, replace=False)
    fig, axes = plt.subplots(5, 5, figsize=(7.2, 7.2))
    for axis, index in zip(axes.ravel(), selected):
        axis.imshow(images[index], cmap="gray")
        axis.set_title(FASHION_LABELS[int(labels[index])], fontsize=8)
        axis.set_axis_off()
    fig.suptitle("Fashion-MNIST: randomly selected training images", y=0.995)
    save_figure(fig, output_dir / "fashion_mnist_examples.png")


def write_dataset_summary(
    output_dir: Path,
    energy_x: pd.DataFrame,
    concrete_x: pd.DataFrame,
) -> None:
    summary = pd.DataFrame(
        [
            ["Synthetic heteroscedastic regression", 450, 1, "Regression"],
            ["Synthetic multimodal regression", 600, 1, "Regression"],
            ["Two moons", 600, 2, "Classification"],
            ["UCI Energy Efficiency", len(energy_x), energy_x.shape[1], "Regression"],
            ["UCI Concrete Compressive Strength", len(concrete_x), concrete_x.shape[1], "Regression"],
            ["Fashion-MNIST training split", 60000, 784, "Classification"],
        ],
        columns=["dataset", "displayed_or_total_samples", "input_dimension", "task"],
    )
    summary.to_csv(output_dir / "dataset_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("figures/datasets"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.data_dir.mkdir(parents=True, exist_ok=True)
    set_plot_style()
    rng = np.random.default_rng(args.seed)

    render_heteroscedastic_regression(args.output_dir, rng)
    render_multimodal_regression(args.output_dir, rng)
    render_two_moons(args.output_dir, args.seed)

    energy_x, energy_y, energy_feature, energy_target = load_energy_efficiency()
    render_tabular_regression(
        energy_x,
        energy_y,
        energy_feature,
        energy_target,
        "UCI Energy Efficiency: Heating Load",
        args.output_dir / "uci_energy_efficiency.png",
    )

    concrete_x, concrete_y, concrete_feature, concrete_target = load_concrete()
    render_tabular_regression(
        concrete_x,
        concrete_y,
        concrete_feature,
        concrete_target,
        "UCI Concrete Compressive Strength",
        args.output_dir / "uci_concrete_compressive_strength.png",
    )

    render_fashion_mnist(args.output_dir, args.data_dir / "fashion_mnist", rng)
    write_dataset_summary(args.output_dir, energy_x, concrete_x)
    print("Finished rendering all six datasets.")


if __name__ == "__main__":
    main()
