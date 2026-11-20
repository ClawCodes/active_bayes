"""Run reproducible GP active-learning experiments and export their data.

The data layout is intentionally independent of this code:

    datasets/<simulator>/run_<number>_seed_<seed>/
      raw_dataset.npz
      raw_metadata.json
      active_learning/<acquisition>/active_learning_dataset.npz
      active_learning/<acquisition>/selection_history.csv
      active_learning/<acquisition>/metadata.json

``active_learning_dataset.npz`` uses precisely the observation/selection row
order.  Its ``raw_index`` array joins every row back to ``raw_dataset.npz``.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from linear_operator.settings import max_cg_iterations

from Hessian_Active_Learning_new import (
    DataLoader,
    active_learning,
    compute_loss_mapgp,
    get_model,
    get_simulator,
    loss_function,
    prior_likelihood,
    rmse,
    save_active_learning_dataset,
    save_raw_dataset,
)
from config_new import DEFAULT_CONFIG


os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def _safe_path_component(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._") or "unnamed"


def _load_python_config(path):
    """Load ``DEFAULT_CONFIG`` from a normal Python config module."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    spec = importlib.util.spec_from_file_location("experiment_config", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import configuration file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "DEFAULT_CONFIG", None)
    if not isinstance(config, dict):
        raise ValueError(f"{path} must define DEFAULT_CONFIG as a dictionary.")
    return dict(config), path


def get_parser(defaults=None, config_path=None):
    parser = argparse.ArgumentParser(
        description="Gaussian-process active learning with durable dataset export."
    )
    parser.set_defaults(**(defaults or {}))

    parser.add_argument("--cfg", type=Path, default=config_path,
                        help="Python file defining DEFAULT_CONFIG.")
    parser.add_argument("--al_type")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--outputs", type=int)
    parser.add_argument("--initial_samples", type=int)
    parser.add_argument("--space_filling_design", choices=["lhs", "random"])
    parser.add_argument("--test_samples", type=int)
    parser.add_argument("--transformation_x")
    parser.add_argument("--transformation_y")
    parser.add_argument("--simulator")
    parser.add_argument("--image_path",
                        help="Optional image path for the mountain/tiger simulators.")
    parser.add_argument("--noise_level", type=float)
    parser.add_argument("--model_type")
    parser.add_argument("--covar_type")
    parser.add_argument("--covar_ard", type=_parse_bool)
    parser.add_argument("--covar_prior")
    parser.add_argument("--mean_type")
    parser.add_argument("--noise_prior", type=_parse_bool)
    parser.add_argument("--initial_lr", type=float)
    parser.add_argument("--milestones", nargs="+", type=int)
    parser.add_argument("--n_epochs", type=int)
    parser.add_argument("--n_runs", type=int)
    parser.add_argument("--num_chains", type=int)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--predict_mcmc")
    parser.add_argument("--selection_criteria")
    parser.add_argument("--active_learning_steps", type=int)
    parser.add_argument("--k_samples", type=int)
    parser.add_argument("--repeat_sampling", type=int)
    parser.add_argument("--folder")
    parser.add_argument("--model_name")
    parser.add_argument("--acquisition_functions", nargs="+")
    parser.add_argument("--kernel_type")

    parser.add_argument("--datasets_dir")
    parser.add_argument("--save_datasets", type=_parse_bool)
    parser.add_argument("--save_every", type=int)
    parser.add_argument("--write_csv", type=_parse_bool)
    parser.add_argument("--max_csv_rows", type=int)
    parser.add_argument("--raw_export_chunk_size", type=int)
    parser.add_argument("--export_test_data", type=_parse_bool)

    parser.add_argument("--topology_image_path")
    parser.add_argument("--topology_persistence_path")
    parser.add_argument("--topology_top_k", type=int)
    return parser


def parse_args(argv=None):
    if argv is None:
        argv = sys.argv[1:]
        if any("ipykernel" in argument or "jupyter" in argument for argument in argv):
            argv = []

    default_config_path = Path(__file__).with_name("config.py")
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--cfg", type=Path)
    preliminary, _ = bootstrap.parse_known_args(argv)

    defaults = dict(DEFAULT_CONFIG)
    config_path = default_config_path
    if preliminary.cfg is not None:
        loaded_defaults, config_path = _load_python_config(preliminary.cfg)
        defaults.update(loaded_defaults)

    parser = get_parser(defaults=defaults, config_path=config_path)
    args = parser.parse_args(argv)
    args.cfg = str(config_path)

    positive_names = (
        "initial_samples", "test_samples", "n_epochs", "n_runs", "num_chains",
        "num_samples", "k_samples", "repeat_sampling",
        "save_every", "max_csv_rows", "raw_export_chunk_size",
    )
    for name in positive_names:
        value = getattr(args, name)
        if value is None or value < 1:
            parser.error(f"--{name} must be at least 1.")
    if args.active_learning_steps is None or args.active_learning_steps < 0:
        parser.error("--active_learning_steps must be zero or greater.")
    if not args.acquisition_functions:
        parser.error("--acquisition_functions must contain at least one acquisition name.")
    return args


def _to_float(value):
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().reshape(-1).mean().item())
    return float(value)


def _dataset_run_dir(args, run, run_seed):
    return (
        Path(args.datasets_dir)
        / _safe_path_component(args.simulator)
        / f"run_{run:03d}_seed_{run_seed}"
    )


def do_active_learning(args):
    simulator = get_simulator(args) if args.simulator else None
    if simulator is None:
        raise ValueError("A simulator is required to run this active-learning pipeline.")

    total_runtime_start = time.time()
    results = {}
    raw_exported_for_run = set()

    for acquisition_name in args.acquisition_functions:
        print(f"\n### Acquisition function: {acquisition_name} ###")
        args.selection_criteria = acquisition_name
        all_nmll_train, all_nmll_valid = [], []
        all_rmse_train, all_rmse_valid = [], []
        dataset_paths = set()

        for run in range(args.n_runs):
            run_seed = args.seed + run
            print(f"\n--- Run {run + 1}/{args.n_runs} (seed={run_seed}) ---")
            random.seed(run_seed)
            np.random.seed(run_seed)
            torch.manual_seed(run_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(run_seed)

            data = DataLoader(args=args, oracle=simulator)
            run_dir = _dataset_run_dir(args, run, run_seed)
            if args.save_datasets and run not in raw_exported_for_run:
                save_raw_dataset(data, run_dir, args, run_seed)
                raw_exported_for_run.add(run)

            nmll_losses, nmll_losses_valid = [], []
            rmse_losses_train, rmse_losses_valid = [], []

            for iteration in range(args.active_learning_steps):
                data.transformed_data_loader()
                model = get_model(args, data, likelihood=prior_likelihood(args))
                nmll_loss, _ = model.fit(
                    (data.train_trans.x, data.train_trans.y), args, debug=False
                )
                nmll_losses.append(_to_float(nmll_loss))

                with max_cg_iterations(5000):
                    predict_output = model.prediction(args, data.test_trans.x)
                predictions_mll = predict_output["predictions"]
                predict_output_train = model.prediction(args, data.train_trans.x)
                predictions_train = predict_output_train["predictions"]
                train_rmse = rmse(
                    predictions_train.mean.view(data.train_trans.y.shape), data.train_trans.y
                ).item()

                losses_valid = compute_loss_mapgp(
                    args,
                    (data.test_trans.x, data.test_trans.y),
                    predictions_mll,
                    lst_metrics=["mll", "rmse"],
                    mll=loss_function(model),
                )
                nmll_losses_valid.append(_to_float(losses_valid["nmll"]))
                rmse_losses_valid.append(_to_float(losses_valid["rmse"]))
                rmse_losses_train.append(_to_float(train_rmse))

                _, selected_x, data = active_learning(
                    args, data, model, predictions_mll, iteration=iteration
                )

                if args.save_datasets and (iteration + 1) % args.save_every == 0:
                    active_dir = save_active_learning_dataset(
                        data=data,
                        run_dir=run_dir,
                        args=args,
                        acquisition_name=acquisition_name,
                        run_number=run,
                        run_seed=run_seed,
                        completed_steps=iteration + 1,
                    )
                    dataset_paths.add(str(active_dir))

                print(
                    f"[Iter {iteration + 1}] selected={selected_x.detach().cpu().numpy().tolist()} "
                    f"Train RMSE: {train_rmse:.4f}, "
                    f"Val RMSE: {_to_float(losses_valid['rmse']):.4f}, "
                    f"NMLL: {_to_float(nmll_loss):.4f}"
                )

            if args.save_datasets:
                active_dir = save_active_learning_dataset(
                    data=data,
                    run_dir=run_dir,
                    args=args,
                    acquisition_name=acquisition_name,
                    run_number=run,
                    run_seed=run_seed,
                    completed_steps=args.active_learning_steps,
                )
                dataset_paths.add(str(active_dir))

            all_nmll_train.append(nmll_losses)
            all_nmll_valid.append(nmll_losses_valid)
            all_rmse_train.append(rmse_losses_train)
            all_rmse_valid.append(rmse_losses_valid)

        results[acquisition_name] = {
            "all_nmll_train": all_nmll_train,
            "all_nmll_valid": all_nmll_valid,
            "all_rmse_train": all_rmse_train,
            "all_rmse_valid": all_rmse_valid,
            "dataset_directories": sorted(dataset_paths),
        }

    total_runtime = time.time() - total_runtime_start
    output_dir = Path(args.folder)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{args.model_name}.pkl"
    with result_path.open("wb") as handle:
        pickle.dump(results, handle, protocol=4)
    print(f"\nSaved results to {result_path} (total runtime: {total_runtime:.2f} sec).")
    return results


def plot_with_variance(all_runs, label, color, marker):
    if not all_runs:
        return
    values = np.asarray(all_runs, dtype=float)
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    iterations = np.arange(1, len(mean) + 1)
    plt.plot(iterations, mean, label=label, color=color, marker=marker)
    plt.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.2)


def save_summary_plots(args, results):
    colors = ["blue", "red", "green", "orange", "purple", "brown"]
    markers = ["o", "x", "s", "d", "^", "v"]
    results_dir = Path(args.folder) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    for metric_key, ylabel, suffix in (
        ("all_nmll_valid", "NMLL", "nmll"),
        ("all_rmse_valid", "RMSE", "rmse"),
    ):
        plt.figure(figsize=(8, 5))
        for index, (acquisition_name, result) in enumerate(results.items()):
            plot_with_variance(
                result[metric_key], acquisition_name,
                colors[index % len(colors)], markers[index % len(markers)],
            )
        plt.xlabel("Active-learning iteration")
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        output_path = results_dir / f"{args.model_name}_{suffix}.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        print(f"Saved {ylabel} plot to {output_path}")


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    results = do_active_learning(args)
    save_summary_plots(args, results)


if __name__ == "__main__":
    main()
