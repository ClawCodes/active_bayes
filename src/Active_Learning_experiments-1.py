
import numpy as np
import torch
import jsonargparse
import pickle
import time
import os
import sys
import matplotlib.pyplot as plt

from Hessian_Active_Learning import DataLoader
from Hessian_Active_Learning import get_simulator
from Hessian_Active_Learning import active_learning
from Hessian_Active_Learning import get_model, compute_loss_mapgp, prior_likelihood, loss_function
from linear_operator.settings import max_cg_iterations
from Hessian_Active_Learning import rmse
import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Limit threading
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def get_parser():
    parser = jsonargparse.ArgumentParser()
    parser.add_argument('--cfg', action=jsonargparse.ActionConfigFile)
    parser.add_argument('--al_type')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--outputs')
    parser.add_argument('--initial_samples')
    parser.add_argument('--space_filling_design')
    parser.add_argument('--test_samples')
    parser.add_argument('--transformation_x')
    parser.add_argument('--transformation_y')
    parser.add_argument('--simulator')
    parser.add_argument('--noise_level')
    parser.add_argument('--model_type')
    parser.add_argument('--covar_type', type=str, default="RBF")
    parser.add_argument('--covar_ard')
    parser.add_argument('--covar_prior')
    parser.add_argument('--mean_type', type=str, default="zero")
    parser.add_argument('--noise_prior')
    parser.add_argument('--initial_lr')
    parser.add_argument('--milestones')
    parser.add_argument('--n_epochs')
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--num_chains')
    parser.add_argument('--num_samples')
    parser.add_argument('--warmup_steps')
    parser.add_argument('--predict_mcmc')
    parser.add_argument('--selection_criteria')
    parser.add_argument('--active_learning_steps', type=int, default=60)
    parser.add_argument('--k_samples', type=int, default=1)
    parser.add_argument('--repeat_sampling', type=int, default=1)
    parser.add_argument('--folder')
    parser.add_argument('--model_name')
    parser.add_argument('--acquisition_functions', nargs='+')
    parser.add_argument('--kernel_type')
    return parser

def do_active_learning(args):
    simulator = get_simulator(args) if args.simulator else None
    total_runtime_start = time.time()

    results = {}

    for acq in args.acquisition_functions:
        print(f"\n### Acquisition Function: {acq} ###")
        args.selection_criteria = acq
        print("HERE1")
        print(args.selection_criteria)

        # Storage for this acquisition function
        all_nmll_train, all_nmll_valid = [], []
        all_rmse_train, all_rmse_valid = [], []

        for run in range(args.n_runs):
            print(f"\n--- Run {run+1}/{args.n_runs} ---")
            np.random.seed(args.seed + run)
            torch.manual_seed(args.seed + run)

            data = DataLoader(args=args, oracle=simulator)

            nmll_losses, nmll_losses_valid = [], []
            rmse_losses_train, rmse_losses_valid = [], []

            for i in range(args.active_learning_steps):
                data.transformed_data_loader()
#                 X_train = data.train_trans.x.detach().cpu().numpy().T
#                 print("shape1")
#                 print(X_train.shape)

                model = get_model(args, data, likelihood=prior_likelihood(args))
                nmll_loss, fit_losses = model.fit((data.train_trans.x, data.train_trans.y), args, False)
                nmll_losses.append(nmll_loss)

                with max_cg_iterations(5000):
                    predict_output = model.prediction(args, data.test_trans.x)
                predictions_mll = predict_output['predictions']

                # Compute RMSE on training set
                predict_output_train = model.prediction(args, data.train_trans.x)
                predictions_train = predict_output_train['predictions']
                
#                 print("shape")
#                 print(data.train_trans.y.shape)
                train_rmse = rmse(predictions_train.mean.view(data.train_trans.y.shape), data.train_trans.y).item()

                losses_valid = compute_loss_mapgp(
                    args,
                    (data.test_trans.x, data.test_trans.y),
                    predictions_mll,
                    lst_metrics=['mll', 'rmse'],
                    mll=loss_function(model)
                )
                nmll_losses_valid.append(losses_valid['nmll'])
                rmse_losses_valid.append(losses_valid['rmse'])
                rmse_losses_train.append(train_rmse)

                # Acquisition
                _, x_new, new_data = active_learning(args, data, model, predictions_mll)

                print(f"[Iter {i+1}] Train RMSE: {train_rmse:.4f}, "
                      f"Val RMSE: {losses_valid['rmse']:.4f}, NMLL: {nmll_loss:.4f}")

            all_nmll_train.append(nmll_losses)
            all_nmll_valid.append(nmll_losses_valid)
            all_rmse_train.append(rmse_losses_train)
            all_rmse_valid.append(rmse_losses_valid)
        
        print("HERE2")
        print(acq)
        results[acq] = {
            "all_nmll_train": all_nmll_train,
            "all_nmll_valid": all_nmll_valid,
            "all_rmse_train": all_rmse_train,
            "all_rmse_valid": all_rmse_valid
        }
        
#         print(results)

    total_runtime = time.time() - total_runtime_start
    os.makedirs(args.folder, exist_ok=True)
    with open(os.path.join(args.folder, f"{args.model_name}.pkl"), 'wb') as fp:
        pickle.dump(results, fp, protocol=4)

    print(f"\nSaved results with {args.n_runs} runs to {args.folder}{args.model_name}.pkl "
          f"(Total runtime: {total_runtime:.2f} sec)")
    
    
# ------------------ PLOTTING ------------------
# def plot_with_variance(all_runs, label, color, marker):
#     if len(all_runs) == 0:
#         return
#     arr = np.array(all_runs)
#     mean = arr.mean(axis=0)
#     std = arr.std(axis=0)
#     iterations = np.arange(1, len(mean)+1)

#     plt.plot(iterations, mean, label=label, color=color, marker=marker)
#     plt.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.2)

def plot_with_variance(all_runs, label, color, marker):
    if len(all_runs) == 0:
        return

    # Convert all elements to CPU numpy arrays
    arr_list = []
    for run in all_runs:
        run_list = []
        for val in run:
            if isinstance(val, torch.Tensor):
                val_cpu = val.detach().cpu()
                # Convert scalar tensors to float
                if val_cpu.ndim == 0:
                    val_cpu = val_cpu.item()
                else:
                    val_cpu = val_cpu.numpy()
                run_list.append(val_cpu)
            else:
                run_list.append(val)
        arr_list.append(run_list)

    arr = np.array(arr_list, dtype=float)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    iterations = np.arange(1, len(mean)+1)

    plt.plot(iterations, mean, label=label, color=color, marker=marker)
    plt.fill_between(iterations, mean - std, mean + std, color=color, alpha=0.2)


# Colors/markers for plotting
colors = ["blue", "red", "green", "orange", "purple", "brown"]
markers = ["o", "x", "s", "d", "^", "v"]

if __name__ == '__main__':
    parser = get_parser()
    if any('ipykernel_launcher' in a or 'jupyter' in a for a in sys.argv):
        args = parser.parse_args(args=['--cfg', 'config.py'])
    else:
        args = parser.parse_args()

    do_active_learning(args)

    # -------- Load results --------
    with open(os.path.join(args.folder, f"{args.model_name}.pkl"), "rb") as f:
        results = pickle.load(f)

    # Create results directory inside args.folder
    results_dir = os.path.join(args.folder, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -------- Validation NMLL --------
    plt.figure(figsize=(8,5))
    for idx, (acq, data) in enumerate(results.items()):
        plot_with_variance(data["all_nmll_valid"], f"{acq}",
                           colors[idx % len(colors)], markers[idx % len(markers)])
    plt.xlabel("")
    plt.ylabel("NMLL")
    plt.title("")
    plt.legend()
    plt.grid(True)
    nmll_path = os.path.join(results_dir, f"{args.model_name}_nmll.png")
    plt.savefig(nmll_path, bbox_inches="tight")
    plt.close()
    print(f"Saved NMLL plot to {nmll_path}")

    # -------- Validation RMSE --------
    plt.figure(figsize=(8,5))
    for idx, (acq, data) in enumerate(results.items()):
        plot_with_variance(data["all_rmse_valid"], f"{acq}",
                           colors[idx % len(colors)], markers[idx % len(markers)])
    plt.xlabel("")
    plt.ylabel("RMSE")
    plt.title("")
    plt.legend()
    plt.grid(True)
    rmse_path = os.path.join(results_dir, f"{args.model_name}_rmse.png")
    plt.savefig(rmse_path, bbox_inches="tight")
    plt.close()
    print(f"Saved RMSE plot to {rmse_path}")