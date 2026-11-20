#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[7]:


import torch
import numpy as np
import csv
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence
import pyro
from pyro.infer.mcmc import NUTS
from pyro.infer.mcmc.api import MCMC
import gpytorch
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
import abc
from scipy import integrate
import torch.nn.functional as F
import skopt
import pandas as pd
from sklearn.neighbors import KernelDensity
from PIL import Image


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# In[9]:


def find_nearest_maxima_index(
    x_query,
    x_locations
):
    """
    x_query:
        (n_query, dim) continuous point(s)

    x_locations:
        (N_maxima, dim)

    Returns:
        one index per query point
    """

    x_locations = torch.as_tensor(x_locations)
    x_query = torch.as_tensor(
        x_query, dtype=x_locations.dtype, device=x_locations.device
    ).reshape(-1, x_locations.shape[1])
    diff = x_query[:, None, :] - x_locations[None, :, :]
    dist_sq = torch.sum(diff ** 2, dim=-1)
    return torch.argmin(dist_sq, dim=1)

def find_local_maxima(Z):
    """
    Z : 2D array

    Returns:
        maxima_indices : list of (row, col)
    """

    rows, cols = Z.shape

    maxima = []

    for i in range(1, rows-1):
        for j in range(1, cols-1):

            val = Z[i, j]

            neighborhood = Z[i-1:i+2, j-1:j+2]

            if val == np.max(neighborhood):

                if np.sum(neighborhood == val) == 1:

                    maxima.append((i, j))

    return maxima


def load_persistence_pairs(csv_path):

    df = pd.read_csv(csv_path)

    birth = df["birth"].values
    death = df["death"].values

    persistence = death - birth

    return birth, death, persistence

def compute_persistence_x(
    Z,
    maxima_indices,
    birth,
    death
):
    """
    persistence(x) for each local maxima
    """

    persistence_x = []

    for (i, j) in maxima_indices:

        z_val = Z[i, j]

        mask = (birth <= z_val) & (z_val <= death)

        if np.any(mask):

            pers = np.max(death[mask] - birth[mask])

        else:

            pers = 0.0

        persistence_x.append(pers)

    return np.array(persistence_x)


def compute_critical_locations(
    Z,
    birth,
    death,
    top_k=50
):
    """
    Select top persistence features
    and assign spatial locations.
    """

    persistence = death - birth

    idx = np.argsort(persistence)[::-1][:top_k]

    critical_locations = []
    persistence_c = []

    rows, cols = Z.shape

    for k in idx:

        d_val = death[k]

        diff = np.abs(Z - d_val)

        i, j = np.unravel_index(
            np.argmin(diff),
            Z.shape
        )

        critical_locations.append((i, j))

        persistence_c.append(persistence[k])

    return (
        np.array(critical_locations),
        np.array(persistence_c)
    )


def assign_nearest_critical(
    maxima_indices,
    critical_locations,
    persistence_c_global
):

    maxima = np.array(maxima_indices)

    c_locations = []
    persistence_c = []

    for x in maxima:

        diff = critical_locations - x

        dist_sq = np.sum(diff**2, axis=1)

        idx = np.argmin(dist_sq)

        c_locations.append(
            critical_locations[idx]
        )

        persistence_c.append(
            persistence_c_global[idx]
        )

    return (
        np.array(c_locations),
        np.array(persistence_c)
    )


def build_x_locations(maxima_indices, x_grid, y_grid):

    x_locations = []

    for (i, j) in maxima_indices:

        x_coord = x_grid[j]
        y_coord = y_grid[i]

        x_locations.append([x_coord, y_coord])

    return np.array(x_locations)


# In[10]:


# Topology data is optional.  It used to be read while importing this module,
# which made *every* experiment depend on the mountain image and its persistence
# CSV, including ordinary sine/branin runs.  Load it only when a topology
# acquisition is actually requested.
_TOPOLOGY_CACHE: Dict[tuple[str, str], Dict[str, torch.Tensor]] = {}


def load_topology_data(
    image_path: str | Path,
    persistence_csv_path: str | Path,
    top_k: int = 50,
) -> Dict[str, torch.Tensor]:
    """Load the image-specific quantities used by the topology acquisitions.

    The returned tensors are cached by absolute input paths, so repeated runs do
    not reload the same image or CSV.  Nothing calls this function unless a
    topology acquisition is selected.
    """
    image_path = Path(image_path).expanduser().resolve()
    persistence_csv_path = Path(persistence_csv_path).expanduser().resolve()
    key = (str(image_path), str(persistence_csv_path))
    if key in _TOPOLOGY_CACHE:
        return _TOPOLOGY_CACHE[key]

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Topology acquisition needs image_path={image_path}, but it was not found."
        )
    if not persistence_csv_path.is_file():
        raise FileNotFoundError(
            "Topology acquisition needs persistence_csv_path="
            f"{persistence_csv_path}, but it was not found."
        )

    image = Image.open(image_path).convert("L")
    z_values = np.asarray(image, dtype=np.float64)
    height, width = z_values.shape
    maxima_indices = find_local_maxima(z_values)
    birth, death, _ = load_persistence_pairs(persistence_csv_path)

    persistence_x_np = compute_persistence_x(
        z_values, maxima_indices, birth, death
    )
    critical_locations, persistence_c_global = compute_critical_locations(
        z_values, birth, death, top_k=top_k
    )
    c_locations_np, persistence_c_np = assign_nearest_critical(
        maxima_indices, critical_locations, persistence_c_global
    )
    x_locations_np = build_x_locations(
        maxima_indices, np.arange(width), np.arange(height)
    )

    result = {
        "persistence_x": torch.as_tensor(persistence_x_np, dtype=torch.float32),
        "x_locations": torch.as_tensor(x_locations_np, dtype=torch.float32),
        "c_locations": torch.as_tensor(c_locations_np, dtype=torch.float32),
        "persistence_c": torch.as_tensor(persistence_c_np, dtype=torch.float32),
    }
    _TOPOLOGY_CACHE[key] = result
    return result


# In[11]:


# #device and importing K_full
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# %store -r K_full
# K_full = K_full.to(device)
# print("K_full shape:", K_full.shape, "device:", K_full.device)


# # **HELPER FUNCTIONS**

# In[12]:


def transform(x, method, mean=None, stddev=None, inverse=False):
    epsilon = 1e-8
#     x = x.to(device)
    
    if method == 'standardize': 
        if inverse:
            return (x * stddev) + mean, mean, stddev
        
        if mean is None:
#             mean = torch.mean(x, dim=0).to(device)
            mean = torch.mean(x, dim=0)
        if stddev is None:
#             stddev = torch.std(x, dim=0).to(device)
            stddev = torch.std(x, dim=0)
        x = (x - mean) / (stddev + epsilon)
        
    elif method == 'identity':
        mean = torch.tensor(0.0, device=device)
        stddev = torch.tensor(1.0, device=device)
        
    elif method == 'min_max_feature_scaling': # transform the data within [0, 1]
        if inverse:
            x = (x * stddev) + mean
            return x, mean, stddev
        
        if mean is None:
            mean = torch.min(x, dim=0)[0]
#             mean = torch.min(x, dim=0)[0].to(device)
            
        if stddev is None:
            stddev = (torch.max(x, dim=0)[0] - mean)
#             stddev = (torch.max(x, dim=0)[0] - mean).to(device)
        x = (x-mean) / (stddev + epsilon)
    
    elif method == 'minusone_one_feature_scaling': # transform the data within the interval [-1, 1]
        if inverse:
            return (x + 1) / 2 * stddev + mean, mean, stddev
        
        if mean is None:
             mean = torch.min(x, dim=0)[0]
#             mean = torch.min(x, dim=0)[0].to(device)
            
        if stddev is None:
            stddev = (torch.max(x, dim=0)[0] - mean)
#             stddev = (torch.max(x, dim=0)[0] - mean).to(device)
        x = 2 * (x - mean) / (stddev + epsilon) - 1
        
    return x, mean, stddev

def prior_likelihood(args):
    if args.covar_prior == "lalprior":
        noise_prior = gpytorch.priors.LogNormalPrior(loc=0, scale=1.73).to(device)
    elif args.covar_prior == "boprior":
        noise_prior = gpytorch.priors.GammaPrior(1.1, 0.05).to(device)
    else:
        return

    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_prior=noise_prior,
        noise_constraint=gpytorch.constraints.Positive()
    ).to(device)

    likelihood.noise = torch.tensor(0.1, device=device)
    return likelihood

def loss_function(model):
    return gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model).to(device)

def optimizer_scheduler(args, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)
    scheduler = None
    if args.milestones is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=100)

    return optimizer, scheduler


# # **DATA**

# In[13]:


class Dataset:
    def __init__(self, x, y):
        if x is not None:
#             self.x = x.to(device)
            self.x = x
        else:
            self.x = x
            
        if y is not None:
#             self.y = y.to(device='cpu')
            self.y = y
        else:
            self.y = y
        
class DataLoader:
    """Data and provenance holder for one active-learning run.

    ``train`` is deliberately stored in observation order.  Its first rows are
    the initial design and every later row is appended at the moment the oracle
    is queried.  The parallel ``selection_*`` lists make that order explicit
    when the run is written to disk.
    """

    def __init__(self, args, oracle):
        self.args = args
        self.oracle = oracle
        # Image grids are naturally integer-valued, but the GP and acquisition
        # optimisers require floating-point coordinates.  Values remain exact
        # pixel coordinates after conversion and are exported as such.
        self.search_space = self.oracle.search_space().to(dtype=torch.float32)

        if self.search_space.ndim != 2:
            raise ValueError(
                "oracle.search_space() must have shape (n_candidates, n_features)."
            )
        if self.search_space.shape[0] < self.args.initial_samples:
            raise ValueError(
                "initial_samples exceeds the number of available raw candidates."
            )

        # The initial design is snapped to this *same* saved candidate pool.
        # Therefore every observed row can be joined to raw_dataset.npz using
        # its raw_index rather than relying on an implicit simulator call.
        initial_indices = self._initial_pool_indices(
            n_samples=self.args.initial_samples,
            space_filling_design=self.args.space_filling_design,
        )
        initial_x = self.search_space[initial_indices]
        initial_y = _ensure_2d(self.oracle.query(initial_x))
        self.train = Dataset(initial_x.detach().clone(), initial_y.detach().clone())

        # Test points remain an independent space-filling design.  They are
        # exported separately and never appear in the raw candidate pool.
        test_x = torch.as_tensor(
            sampling_initial_points(
                self.args.test_samples,
                self.search_space.detach().cpu().numpy(),
                space_filling_design="lhs",
            ),
            dtype=torch.float32,
            device=self.oracle.device,
        )
        if test_x.ndim == 1:
            test_x = test_x.unsqueeze(-1)
        test_y = _ensure_2d(self.oracle.query(test_x))
        self.test = Dataset(test_x, test_y)

        self.true_mean = _ensure_2d(self.oracle.mean(self.search_space)).detach().clone()
        self.true_std = _ensure_2d(self.oracle.stddev(self.search_space)).detach().clone()
        self.pool_labeled = self.train.x.detach().clone()
        self.oracle_labels = None
        self.true_mean_cp = None
        self.true_std_cp = None

        self.observed_raw_indices = set(int(i) for i in initial_indices.detach().cpu().tolist())
        self.selection_steps = [-1] * int(self.train.x.shape[0])
        self.selection_ranks = list(range(int(self.train.x.shape[0])))
        self.repeat_ids = [0] * int(self.train.x.shape[0])
        self.acquisition_scores = [float("nan")] * int(self.train.x.shape[0])
        self.raw_indices = [int(i) for i in initial_indices.detach().cpu().tolist()]
        self.is_initial = [True] * int(self.train.x.shape[0])

        self.train_trans = Dataset(None, None)
        self.test_trans = Dataset(None, None)

        self.gp_train_loader_ori = None
        self.gp_test_loader_ori = None
        self.gp_train_loader = None
        self.gp_test_loader = None

        self.x_mean, self.x_stddev = None, None
        self.y_mean, self.y_stddev = None, None

    def _initial_pool_indices(self, n_samples, space_filling_design):
        """Return unique raw-pool rows nearest to a space-filling design."""
        proposed = sampling_initial_points(
            n_samples,
            self.search_space.detach().cpu().numpy(),
            space_filling_design=space_filling_design,
        )
        proposed = torch.as_tensor(
            proposed, dtype=self.search_space.dtype, device=self.search_space.device
        )
        if proposed.ndim == 1:
            proposed = proposed.unsqueeze(-1)

        available = torch.ones(self.search_space.shape[0], dtype=torch.bool, device=self.search_space.device)
        chosen = []
        for point in proposed:
            distances = torch.sum((self.search_space - point) ** 2, dim=1)
            distances = distances.masked_fill(~available, float("inf"))
            index = int(torch.argmin(distances).item())
            if not torch.isfinite(distances[index]):
                raise RuntimeError("Unable to choose a unique initial raw-data row.")
            available[index] = False
            chosen.append(index)
        return torch.as_tensor(chosen, dtype=torch.long, device=self.search_space.device)

    def select_available_pool_points(self, point, k_samples=1):
        """Snap an optimizer proposal to unique, as-yet-unobserved raw rows."""
        point = torch.as_tensor(
            point, dtype=self.search_space.dtype, device=self.search_space.device
        ).reshape(1, -1)
        if point.shape[1] != self.search_space.shape[1]:
            raise ValueError("Candidate dimension does not match the raw candidate pool.")

        if len(self.observed_raw_indices) + k_samples > self.search_space.shape[0]:
            raise RuntimeError("The active-learning candidate pool has been exhausted.")

        distances = torch.sum((self.search_space - point) ** 2, dim=1)
        if self.observed_raw_indices:
            used = torch.as_tensor(
                sorted(self.observed_raw_indices),
                dtype=torch.long,
                device=self.search_space.device,
            )
            distances[used] = float("inf")

        raw_indices = torch.topk(distances, k=k_samples, largest=False).indices
        if not torch.all(torch.isfinite(distances[raw_indices])):
            raise RuntimeError("No unobserved raw candidate is available for selection.")
        return self.search_space[raw_indices], raw_indices

    def record_observations(
        self,
        x,
        y,
        raw_indices,
        iteration,
        acquisition_scores,
        repeat_sampling,
    ):
        """Append queried rows and provenance in exactly the same order."""
        x = torch.as_tensor(x, dtype=self.train.x.dtype, device=self.train.x.device)
        if x.ndim == 1:
            x = x.unsqueeze(-1)
        y = _ensure_2d(y).to(self.train.y.device, dtype=self.train.y.dtype)
        raw_indices = torch.as_tensor(raw_indices, dtype=torch.long).reshape(-1)
        scores = torch.as_tensor(acquisition_scores, dtype=torch.float64).reshape(-1)

        if x.shape[0] != y.shape[0] or x.shape[0] != raw_indices.shape[0]:
            raise ValueError("x, y, and raw_indices must have the same number of selected rows.")
        if scores.numel() == 1 and x.shape[0] > 1:
            scores = scores.repeat(x.shape[0])
        if scores.numel() != x.shape[0]:
            raise ValueError("acquisition_scores must be scalar or one value per selected row.")

        # repeat_sampling means repeated observations of the same selected raw
        # rows.  The ordering is [rank 0, ..., rank k-1] for each repeat.
        appended_x = x.repeat((repeat_sampling, 1))
        appended_y = y.repeat((repeat_sampling, 1))
        appended_indices = raw_indices.repeat(repeat_sampling)
        appended_scores = scores.repeat(repeat_sampling)

        self.train.x = torch.cat([self.train.x, appended_x], dim=0)
        self.train.y = torch.cat([_ensure_2d(self.train.y), appended_y], dim=0)
        self.pool_labeled = torch.cat([self.pool_labeled, appended_x], dim=0)
        self.observed_raw_indices.update(int(i) for i in raw_indices.detach().cpu().tolist())

        for repeat_id in range(repeat_sampling):
            for rank, (raw_index, score) in enumerate(zip(raw_indices.tolist(), scores.tolist())):
                self.selection_steps.append(int(iteration))
                self.selection_ranks.append(int(rank))
                self.repeat_ids.append(int(repeat_id))
                self.acquisition_scores.append(float(score))
                self.raw_indices.append(int(raw_index))
                self.is_initial.append(False)
        
    def transformed_data_loader(self):
        train_x_trans, self.x_mean, self.x_stddev = transform(
            self.train.x, self.args.transformation_x
        )
        train_y_trans, self.y_mean, self.y_stddev = transform(
            self.train.y, self.args.transformation_y
        )
        
        test_x_trans, _, _ = transform(self.test.x, self.args.transformation_x, self.x_mean, self.x_stddev)
        test_y_trans, _, _ = transform(self.test.y, self.args.transformation_y, self.y_mean, self.y_stddev)

        train_y_trans = train_y_trans.squeeze(-1)
        test_y_trans = test_y_trans.squeeze(-1)

        self.train_trans = Dataset(train_x_trans, train_y_trans)
        self.test_trans = Dataset(test_x_trans, test_y_trans)
        
        self.gp_train_loader = (self.train_trans.x, self.train_trans.y)
        self.gp_test_loader = (self.test_trans.x, self.test_trans.y)
        
        self.gp_train_loader_ori = (self.train.x, self.train.y)
        self.gp_test_loader_ori = (self.test.x, self.test.y)
        
        return
    
def _ensure_2d(values):
    """Represent scalar targets as (n_rows, 1) without changing row order."""
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    if values.ndim == 0:
        return values.reshape(1, 1)
    if values.ndim == 1:
        return values.unsqueeze(-1)
    return values


def sampling_initial_points(n_samples, search_space, space_filling_design="lhs"):
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.sampler.Lhs.html#skopt.sampler.Lhs
    if search_space.ndim == 1:
        bounds = [(np.min(search_space), np.max(search_space))]
    else:
        mins = np.min(search_space, axis=0)
        maxs = np.max(search_space, axis=0)
        bounds = [(low, high) for low, high in zip(mins, maxs)]

    if space_filling_design == "lhs":
        space = skopt.space.Space(bounds)
        lhs = skopt.sampler.Lhs(lhs_type="classic", criterion="maximin")
        x_sample = np.array(lhs.generate(space.dimensions, n_samples))
    elif space_filling_design == "random":
        lower = np.asarray([bound[0] for bound in bounds])
        upper = np.asarray([bound[1] for bound in bounds])
        x_sample = np.random.uniform(lower, upper, size=(n_samples, len(bounds)))
    else:
        raise ValueError(
            "space_filling_design must be 'lhs' or 'random', "
            f"got {space_filling_design!r}."
        )

    if search_space.ndim == 1:
        x_sample = x_sample.squeeze(-1)

    return x_sample


# # **SIMULATORS**

# In[14]:


##citation

class Simulator(abc.ABC):
    def __init__(self, n_points, noise_sigma, n_inputs, n_outputs, device=None):
        self.n_points = n_points
        self.noise_sigma = noise_sigma
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def query(self, x):
        if not torch.is_tensor(x):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)
        y = self.mean(x) + self.stddev(x) * self.noise(x)

        # Safely reshape if y is 2D with a single column
        if y.ndim == 2 and y.shape[1] == 1:
            y = y.reshape(-1)

        return y
    
    def sample_initial_data(self, n_samples, space_filling_design):
        x = np.array(sampling_initial_points(n_samples, self.search_space().cpu().numpy()))
        y = self.query(x)
#         x = torch.tensor(x, dtype=torch.float32, device=self.device)
        x = torch.from_numpy(x).float().to(self.device)
        if x.ndim == 1:
            x = x.view(-1, 1)
#         y = torch.tensor(y, dtype=torch.float32, device=self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y).float().to(self.device)
        else:
            y = y.float().to(self.device)
            
        if self.n_outputs == 1:
            y = y.squeeze(-1)

        return x, y
    
    def __call__(self, x):
        return self.query(x)

    @abc.abstractmethod
    def mean(self, x):
        pass

    @abc.abstractmethod
    def stddev(self, x):
        pass

    @abc.abstractmethod
    def noise(self, x):
        pass

    @abc.abstractmethod
    def search_space(self):
        pass

class Sine1d(Simulator):
    def __init__(self, n_points=101, noise_sigma=0.1, n_inputs=1, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

    def mean(self, x):
        x = x.squeeze(-1)
        y = torch.sin(2 * np.pi * x)
        return y.unsqueeze(-1)

    def stddev(self, x):
        return self.noise_sigma * torch.ones_like(x)

    def noise(self, x):
        return torch.normal(mean=0, std=self.noise_sigma, size=[x.shape[0], self.n_outputs]).to(self.device)

    def search_space(self):
        grid = np.linspace(0, 1, self.n_points).reshape(-1, 1)
        return torch.tensor(grid, dtype=torch.float32, device=self.device)

class Higdon1d(Simulator):
    def __init__(self, n_points=101, noise_sigma=0.1, n_inputs=1, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)
        self.n_inputs = 1
        self.n_outputs = 1

    def mean(self, x):
        x = x.squeeze(-1)
        mean = torch.where(
            x < 10,
            torch.sin(np.pi * x / 5) + 0.2 * torch.cos(4 * np.pi * x / 5),
            x / 10 - 1
        )
        return mean.unsqueeze(-1)

    def stddev(self, x):
        return self.noise_sigma * torch.ones_like(x)

    def noise(self, x):
        return torch.normal(mean=0, std=self.noise_sigma, size=[x.shape[0], self.n_outputs]).to(self.device)

    def search_space(self):
        grid = np.linspace(0, 20, self.n_points).reshape(-1, 1)
        return torch.tensor(grid, dtype=torch.float32, device=self.device)

class GramacyAndLee1d(Simulator):
    def __init__(self, n_points=101, noise_sigma=0.1, n_inputs=1, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)

    def mean(self, x):
        x = x.squeeze(-1)
        val = torch.sin(10 * torch.pi * x) / (2 * x) + (x - 1).pow(4)
        return val.unsqueeze(-1)

    def stddev(self, x):
        return self.noise_sigma * torch.ones_like(x, device=self.device)

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        grid = torch.linspace(0.5, 2.5, self.n_points, device=self.device).unsqueeze(-1)
        return grid


class GramacyAndLee2d(Simulator):
    def __init__(self, n_points=100, noise_sigma=0.05, n_inputs=2, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)

    def mean(self, x):
        mean = x[:, 0] * torch.exp(-(x[:, 0].pow(2) + x[:, 1].pow(2)))
        return mean.unsqueeze(-1)

    def stddev(self, x):
        stddev = self.noise_sigma * torch.ones(x.shape[0], 1, device=self.device)
        return stddev

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        x1 = torch.linspace(-2, 6, self.n_points, device=self.device)
        x2 = torch.linspace(-2, 6, self.n_points, device=self.device)
        grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
        search_space = torch.stack([grid_x1.flatten(), grid_x2.flatten()], dim=-1)
        return search_space

class Branin2d(Simulator):
    def __init__(self, n_points=100, noise_sigma=11.32, n_inputs=2, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)

    def mean(self, x):
        a = 1.0
        b = 5.1 / (4 * torch.pi**2)
        c = 5.0 / torch.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8 * torch.pi)

        mean = (
            a * (x[:, 1] - b * x[:, 0].pow(2) + c * x[:, 0] - r).pow(2)
            + s * (1 - t) * torch.cos(x[:, 0])
            + s
            + 5 * x[:, 0]
        )
        return mean.unsqueeze(-1)

    def stddev(self, x):
        stddev = self.noise_sigma * torch.ones(x.shape[0], 1, device=self.device)
        return stddev

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        x1 = torch.linspace(-5, 10, self.n_points, device=self.device)
        x2 = torch.linspace(0, 15, self.n_points, device=self.device)
        grid_x1, grid_x2 = torch.meshgrid(x1, x2, indexing="ij")
        search_space = torch.stack([grid_x1.flatten(), grid_x2.flatten()], dim=-1)
        return search_space

class Ishigami3d(Simulator):
    def __init__(self, n_points=100, noise_sigma=0.187, n_inputs=3, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)

    def mean(self, x):
        a = 7.0
        b = 0.1
        # ensure float32
        x = x.to(torch.float32)
        mean = torch.sin(x[:, 0]) + a * torch.sin(x[:, 1]).pow(2) + b * x[:, 2].pow(4) * torch.sin(x[:, 0])
        return mean.unsqueeze(-1)

    def stddev(self, x):
        return self.noise_sigma * torch.ones((x.shape[0], 1), device=self.device, dtype=torch.float32)

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        x1 = torch.linspace(-np.pi, np.pi, self.n_points, device=self.device)
        x2 = torch.linspace(-np.pi, np.pi, self.n_points, device=self.device)
        x3 = torch.linspace(-np.pi, np.pi, self.n_points, device=self.device)
        grid = torch.stack(torch.meshgrid(x1, x2, x3, indexing='ij'), dim=-1).reshape(-1, 3)
        return grid.to(torch.float32)

class Hartmann6d(Simulator):
    def __init__(self, n_points=100, noise_sigma=0.0192, n_inputs=6, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)

        # constants as torch tensors
        self.alpha = torch.tensor([1.0, 1.2, 3.0, 3.2], device=self.device)
        self.A = torch.tensor([
            [10, 3, 17, 3.5, 1.7, 8],
            [0.05, 10, 17, 0.1, 8, 14],
            [3, 3.5, 1.7, 10, 17, 8],
            [17, 8, 0.05, 10, 0.1, 14]
        ], device=self.device)

        self.P = 1e-4 * torch.tensor([
            [1312, 1696, 5569, 124, 8283, 5886],
            [2329, 4135, 8307, 3736, 1004, 9991],
            [2348, 1451, 3522, 2883, 3047, 6650],
            [4047, 8828, 8732, 5743, 1091, 381]
        ], device=self.device)

    def mean(self, x):
        x = x.to(torch.float32).to(self.device)  # [batch, 6]

        # expand x for broadcasting: [batch, 1, 6]
        x_exp = x.unsqueeze(1)  # [batch, 1, 6]

        # expand constants: [1, 4, 6]
        A = self.A.unsqueeze(0)  # [1, 4, 6]
        P = self.P.unsqueeze(0)  # [1, 4, 6]
        alpha = self.alpha.unsqueeze(0)  # [1, 4]

        # compute inner sum
        inner = torch.sum(A * (x_exp - P).pow(2), dim=2)  # [batch, 4]

        mean = -torch.sum(alpha * torch.exp(-inner), dim=1)  # [batch]
        return mean.unsqueeze(-1)  # [batch, 1]

    def stddev(self, x):
        return self.noise_sigma * torch.ones((x.shape[0], 1), device=self.device, dtype=torch.float32)

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        # sample uniformly in [0, 1]^6
        return torch.rand((1_000_000, 6), device=self.device, dtype=torch.float32)
    
##citation
# Ackley function, commonly used in global optimization benchmarks
# Reference: D. H. Ackley, "A connectionist machine for genetic hillclimbing", 1987

class Ackley6d(Simulator):
    def __init__(self, n_points=100, noise_sigma=0.1, n_inputs=6, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)
        self.a = 20.0
        self.b = 0.2
        self.c = 2 * np.pi

    def mean(self, x):
        x = x.to(torch.float32).to(self.device)  # [batch, 6]
        d = x.shape[1]

        sum_sq = torch.sum(x.pow(2), dim=1)
        sum_cos = torch.sum(torch.cos(self.c * x), dim=1)

        term1 = -self.a * torch.exp(-self.b * torch.sqrt(sum_sq / d))
        term2 = -torch.exp(sum_cos / d)

        mean = term1 + term2 + self.a + np.e
        return mean.unsqueeze(-1)  # [batch, 1]

    def stddev(self, x):
        return self.noise_sigma * torch.ones((x.shape[0], 1), device=self.device, dtype=torch.float32)

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        # standard Ackley domain [-32.768, 32.768]^6
        return torch.rand((1_000_000, 6), device=self.device) * (32.768 - (-32.768)) + (-32.768)
    
    
##citation
# Rastrigin function, a standard 1D benchmark for optimization
# Reference: M. Rastrigin, "Systems of Extremal Control," 1974

class Rastrigin1d(Simulator):
    def __init__(self, n_points=101, noise_sigma=0.1, n_inputs=1, n_outputs=1, device=None):
        super().__init__(n_points, noise_sigma, n_inputs, n_outputs, device)

    def mean(self, x):
        x = x.squeeze(-1)
        mean = 10 + (x.pow(2) - 10 * torch.cos(2 * np.pi * x))
        return mean.unsqueeze(-1)

    def stddev(self, x):
        return self.noise_sigma * torch.ones_like(x, device=self.device)

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    def search_space(self):
        # Standard Rastrigin 1D domain [-5.12, 5.12]
        grid = torch.linspace(-5.12, 5.12, self.n_points, device=self.device).unsqueeze(-1)
        return grid


class Salomon2D(Simulator):
    """
    Salomon function simulator:

        f(x, y) =
            1 - cos(2πr) + 0.1 r
        where r = sqrt(x^2 + y^2)

    This class replaces the raw Z usage and
    provides a clean GP-compatible interface.
    """

    def __init__(
        self,
        n_points=100,
        noise_sigma=0.0,
        n_inputs=2,
        n_outputs=1,
        device=None,
        xmin=-10,
        xmax=10,
        ymin=-10,
        ymax=10,
    ):
        super().__init__(
            n_points,
            noise_sigma,
            n_inputs,
            n_outputs,
            device
        )

        self.xmin = xmin
        self.xmax = xmax
        self.ymin = ymin
        self.ymax = ymax

    # -------------------------
    # Salomon function
    # -------------------------
    def mean(self, x):
        """
        x: (N, 2)
        """
        x = x.to(self.device)

        x1 = x[:, 0]
        x2 = x[:, 1]

        r = torch.sqrt(x1**2 + x2**2)

        y = 1 - torch.cos(2 * np.pi * r) + 0.1 * r

        return y.unsqueeze(-1)

    # -------------------------
    # Noise model
    # -------------------------
    def stddev(self, x):
        return self.noise_sigma * torch.ones(
            (x.shape[0], 1),
            device=self.device
        )

    def noise(self, x):
        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    # -------------------------
    # Search space (THIS replaces your grid X,Y,Z)
    # -------------------------
    def search_space(self):
        """
        Returns a structured 2D grid of candidate points:
        shape = (n_points^2, 2)
        """

        x = torch.linspace(
            self.xmin,
            self.xmax,
            self.n_points,
            device=self.device
        )

        y = torch.linspace(
            self.ymin,
            self.ymax,
            self.n_points,
            device=self.device
        )

        X, Y = torch.meshgrid(x, y, indexing="ij")

        grid = torch.stack(
            [X.reshape(-1), Y.reshape(-1)],
            dim=-1
        )

        return grid 
    

class TigerImage2D(Simulator):
    """
    Tiger image simulator:

    Treats grayscale pixel intensity as scalar field:
        f(x, y) = pixel intensity

    Domain:
        x ∈ [0, width-1]
        y ∈ [0, height-1]

    Compatible with GP active learning framework.
    """

    def __init__(
        self,
        image_path="tiger-topology.jpg",
        noise_sigma=0.0,
        n_inputs=2,
        n_outputs=1,
        device=None,
    ):
        # Load image
        img = Image.open(image_path).convert("L")

        Z = np.array(img, dtype=np.float64)

        self.Z = torch.tensor(
            Z,
            dtype=torch.float32,
            device=device
        )

        self.height, self.width = Z.shape

        # Total grid points
        n_points = self.width * self.height

        super().__init__(
            n_points,
            noise_sigma,
            n_inputs,
            n_outputs,
            device
        )

    # -------------------------
    # Mean function
    # -------------------------
    def mean(self, x):
        """
        x: (N, 2)
            pixel coordinates

        Returns:
            intensity values
        """

        x = x.to(self.device)

        # Convert to integer pixel indices
        col = torch.clamp(
            x[:, 0].long(),
            0,
            self.width - 1
        )

        row = torch.clamp(
            x[:, 1].long(),
            0,
            self.height - 1
        )

        y = self.Z[row, col]

        return y.unsqueeze(-1)

    # -------------------------
    # Noise model
    # -------------------------
    def stddev(self, x):

        return self.noise_sigma * torch.ones(
            (x.shape[0], 1),
            device=self.device
        )

    def noise(self, x):

        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    # -------------------------
    # Search space
    # -------------------------
    def search_space(self):
        """
        Returns all pixel coordinates:

            shape = (width*height, 2)

        Each point:
            [x_pixel, y_pixel]
        """

        x_coords = torch.arange(
            self.width,
            device=self.device
        )

        y_coords = torch.arange(
            self.height,
            device=self.device
        )

        X, Y = torch.meshgrid(
            x_coords,
            y_coords,
            indexing="xy"
        )

        grid = torch.stack(
            [X.reshape(-1), Y.reshape(-1)],
            dim=-1
        )

        return grid
    
class MountainImage2D(Simulator):
    """
    Tiger image simulator:

    Treats grayscale pixel intensity as scalar field:
        f(x, y) = pixel intensity

    Domain:
        x ∈ [0, width-1]
        y ∈ [0, height-1]

    Compatible with GP active learning framework.
    """

    def __init__(
        self,
        image_path="mt_sugarloof_new_zealand.jpg",
        noise_sigma=0.0,
        n_inputs=2,
        n_outputs=1,
        device=None,
    ):
        # Load image
        img = Image.open(image_path).convert("L")

        Z = np.array(img, dtype=np.float64)

        self.Z = torch.tensor(
            Z,
            dtype=torch.float32,
            device=device
        )

        self.height, self.width = Z.shape

        # Total grid points
        n_points = self.width * self.height

        super().__init__(
            n_points,
            noise_sigma,
            n_inputs,
            n_outputs,
            device
        )

    # -------------------------
    # Mean function
    # -------------------------
    def mean(self, x):
        """
        x: (N, 2)
            pixel coordinates

        Returns:
            intensity values
        """

        x = x.to(self.device)

        # Convert to integer pixel indices
        col = torch.clamp(
            x[:, 0].long(),
            0,
            self.width - 1
        )

        row = torch.clamp(
            x[:, 1].long(),
            0,
            self.height - 1
        )

        y = self.Z[row, col]

        return y.unsqueeze(-1)

    # -------------------------
    # Noise model
    # -------------------------
    def stddev(self, x):

        return self.noise_sigma * torch.ones(
            (x.shape[0], 1),
            device=self.device
        )

    def noise(self, x):

        return torch.normal(
            mean=0.0,
            std=self.noise_sigma,
            size=(x.shape[0], self.n_outputs),
            device=self.device
        )

    # -------------------------
    # Search space
    # -------------------------
    def search_space(self):
        """
        Returns all pixel coordinates:

            shape = (width*height, 2)

        Each point:
            [x_pixel, y_pixel]
        """

        x_coords = torch.arange(
            self.width,
            device=self.device
        )

        y_coords = torch.arange(
            self.height,
            device=self.device
        )

        X, Y = torch.meshgrid(
            x_coords,
            y_coords,
            indexing="xy"
        )

        grid = torch.stack(
            [X.reshape(-1), Y.reshape(-1)],
            dim=-1
        )

        return grid
    
def get_simulator(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simulator = args.simulator
    noise_sigma = float(getattr(args, "noise_level", 0.0))
    if simulator == "gramacy1d":
        return GramacyAndLee1d(noise_sigma=noise_sigma, device=device)
    elif simulator == "higdon1d":
        return Higdon1d(noise_sigma=noise_sigma, device=device)
    elif simulator == 'rastrigin1d':
        return Rastrigin1d(noise_sigma=noise_sigma, device=device)
    elif simulator == "gramacy2d":
        return GramacyAndLee2d(noise_sigma=noise_sigma, device=device)
    elif simulator == "branin2d":
        return Branin2d(noise_sigma=noise_sigma, device=device)
    elif simulator == "ishigami3d":
        return Ishigami3d(noise_sigma=noise_sigma, device=device)
    elif simulator == "hartmann6d":
        return Hartmann6d(noise_sigma=noise_sigma, device=device)
    elif simulator == "sine1d":
        return Sine1d(noise_sigma=noise_sigma, device=device)
    elif simulator == "ackley6d":
        return Ackley6d(noise_sigma=noise_sigma, device=device)
    elif simulator == "salomon2D":
        return Salomon2D(noise_sigma=noise_sigma, device=device)
    elif simulator == "tiger":
        image_path = getattr(args, "image_path", None) or "tiger-topology.jpg"
        return TigerImage2D(image_path=image_path, noise_sigma=noise_sigma, device=device)
    elif simulator == "mountain":
        image_path = getattr(args, "image_path", None) or "mt_sugarloof_new_zealand.jpg"
        return MountainImage2D(image_path=image_path, noise_sigma=noise_sigma, device=device)
    else:
        raise NotImplementedError(f'Sorry, simulator {simulator} is not there for now.')


# ---------------------------------------------------------------------------
# Dataset export
# ---------------------------------------------------------------------------
# Files written by these helpers are intentionally plain NumPy archives plus a
# CSV/JSON sidecar.  A collaborator can load the data without importing this
# module or recreating a simulator object.

DATASET_SCHEMA_VERSION = 1


def _tensor_to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_path_component(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return value or "unnamed"


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return _tensor_to_numpy(value).tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def _config_snapshot(args):
    """Return only JSON-safe configuration values for durable provenance."""
    return {
        key: _json_value(value)
        for key, value in sorted(vars(args).items())
        if not key.startswith("_")
    }


def _write_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(_json_value(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_tabular_csv(path, x, y, extra_columns=None):
    """Write a portable row-wise view of a feature/target data set."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x[:, None]
    if y.ndim == 1:
        y = y[:, None]
    if len(x) != len(y):
        raise ValueError("x and y must contain the same number of rows.")

    extra_columns = extra_columns or {}
    for name, column in extra_columns.items():
        if len(column) != len(x):
            raise ValueError(f"Column {name!r} does not match the number of data rows.")

    header = [f"x_{idx}" for idx in range(x.shape[1])]
    header += [f"y_{idx}" for idx in range(y.shape[1])]
    header += list(extra_columns)

    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for row in range(len(x)):
            values = x[row].tolist() + y[row].tolist()
            values += [np.asarray(column)[row].item() for column in extra_columns.values()]
            writer.writerow(values)


def _oracle_mean_in_chunks(oracle, x, chunk_size):
    """Evaluate a raw target set without constructing one huge autograd graph."""
    targets = []
    with torch.no_grad():
        for start in range(0, x.shape[0], chunk_size):
            targets.append(_ensure_2d(oracle.mean(x[start:start + chunk_size])).detach().cpu())
    return torch.cat(targets, dim=0)


def save_raw_dataset(data, output_dir, args, run_seed):
    """Save the fixed candidate pool and its deterministic, noise-free labels.

    ``raw_dataset.npz`` contains ``x``, ``y``, and ``row_id``.  Every selected
    observation in the matching active-learning export has a ``raw_index`` that
    points to one of these rows.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_x = _tensor_to_numpy(data.search_space).astype(np.float32, copy=False)
    raw_y = _tensor_to_numpy(
        _oracle_mean_in_chunks(
            data.oracle,
            data.search_space,
            chunk_size=int(getattr(args, "raw_export_chunk_size", 65536)),
        )
    ).astype(np.float32, copy=False)
    raw_y = raw_y.reshape(raw_y.shape[0], -1)
    row_id = np.arange(raw_x.shape[0], dtype=np.int64)

    raw_path = output_dir / "raw_dataset.npz"
    np.savez(raw_path, x=raw_x, y=raw_y, row_id=row_id)

    csv_limit = int(getattr(args, "max_csv_rows", 100000))
    wrote_csv = bool(getattr(args, "write_csv", True)) and raw_x.shape[0] <= csv_limit
    if wrote_csv:
        _write_tabular_csv(output_dir / "raw_dataset.csv", raw_x, raw_y, {"row_id": row_id})

    if bool(getattr(args, "export_test_data", True)):
        test_x = _tensor_to_numpy(data.test.x).astype(np.float32, copy=False)
        test_y = _tensor_to_numpy(_ensure_2d(data.test.y)).astype(np.float32, copy=False)
        np.savez(output_dir / "test_dataset.npz", x=test_x, y=test_y)

    _write_json(
        output_dir / "raw_metadata.json",
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "simulator": args.simulator,
            "run_seed": int(run_seed),
            "n_rows": int(raw_x.shape[0]),
            "n_features": int(raw_x.shape[1]),
            "n_targets": int(raw_y.shape[1]),
            "target_source": "oracle.mean (noise-free)",
            "row_id_semantics": "row_id indexes raw_dataset.npz directly",
            "csv_written": wrote_csv,
            "csv_row_limit": csv_limit,
            "config": _config_snapshot(args),
        },
    )
    return raw_path


def save_active_learning_dataset(
    data,
    run_dir,
    args,
    acquisition_name,
    run_number,
    run_seed,
    completed_steps,
):
    """Save the selection-ordered observations for one acquisition/run.

    The rows of ``active_learning_dataset.npz`` and
    ``selection_history.csv`` are identical to the row order in ``data.train``:
    initial design first, then each active query in chronological order.
    """
    active_dir = (
        Path(run_dir)
        / "active_learning"
        / _safe_path_component(acquisition_name)
    )
    active_dir.mkdir(parents=True, exist_ok=True)

    x = _tensor_to_numpy(data.train.x).astype(np.float32, copy=False)
    y = _tensor_to_numpy(_ensure_2d(data.train.y)).astype(np.float32, copy=False)
    n_rows = x.shape[0]

    provenance = {
        "selection_order": np.arange(n_rows, dtype=np.int64),
        "selection_step": np.asarray(data.selection_steps, dtype=np.int64),
        "selection_rank": np.asarray(data.selection_ranks, dtype=np.int64),
        "repeat_id": np.asarray(data.repeat_ids, dtype=np.int64),
        "acquisition_score": np.asarray(data.acquisition_scores, dtype=np.float64),
        "raw_index": np.asarray(data.raw_indices, dtype=np.int64),
        "is_initial": np.asarray(data.is_initial, dtype=np.bool_),
    }
    if any(len(values) != n_rows for values in provenance.values()):
        raise RuntimeError(
            "Selection provenance no longer matches train rows; refusing to write an ambiguous dataset."
        )

    np.savez(active_dir / "active_learning_dataset.npz", x=x, y=y, **provenance)
    _write_tabular_csv(active_dir / "selection_history.csv", x, y, provenance)

    _write_json(
        active_dir / "metadata.json",
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "simulator": args.simulator,
            "acquisition": acquisition_name,
            "run_number": int(run_number),
            "run_seed": int(run_seed),
            "completed_active_learning_steps": int(completed_steps),
            "n_observations": int(n_rows),
            "ordering": (
                "Rows are initial-design observations followed by active observations "
                "in query order; repeat_id distinguishes repeated observations."
            ),
            "raw_join": "raw_index joins raw_dataset.npz[row_id] to each selected x row.",
            "observed_target_source": "oracle.query (may include configured observation noise)",
            "config": _config_snapshot(args),
        },
    )
    return active_dir


# # **METRICS FOR EVALUATION**

# In[15]:


def rmse(pred, target):
    """Root mean square error
    
    pred: torch.tensor
    target: torch.tensor
    """
    pred = pred.to(device)
    target = target.to(device)
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=0))

def mse(pred, target):
    """
    Mean Square Error

    pred: torch.tensor
    target: torch.tensor
    """
    pred = pred.to(device)
    target = target.to(device)
    return torch.mean((pred - target) ** 2, dim=0)

def mae(pred, target):
    """
    Mean Absolute Error

    pred: torch.tensor
    target: torch.tensor
    """
    pred = pred.to(device)
    target = target.to(device)
    return torch.mean(torch.abs(pred - target), dim=0)

def rrse(pred, target):
    """
    Root Relative Squared Error
    (https://www.gepsoft.com/GeneXproTools/AnalysesAndComputations/MeasuresOfFit/RootRelativeSquaredError.htm)
    """
    pred = pred.to(device)
    target = target.to(device)
    return torch.sqrt(mse(pred, target) / mse(torch.mean(target, dim=0), target))


def rae(pred, target):
    """
    Relative Absolute Error

    pred: torch.tensor
    target: torch.tensor
    """
    pred = pred.to(device)
    target = target.to(device)
    return mae(pred, target) / mae(torch.mean(target, dim=0), target)


def picp(pred, ci_lower, ci_upper):
    """
    Prediction Interval Coverage Probability [1]

    [1] A. Khosravi, E. Mazloumi, S. Nahavandi, D. Creighton, and J. W. C. Van Lint,
    “Prediction intervals to account for uncertainties in travel time prediction,”
    IEEE Trans. Intell. Transp. Syst., vol. 12, no. 2, pp. 537–547, Jun. 2011.

    pred: torch.tensor
    ci_lower: torch.tensor, lower confidence interval for the corresponding targets
    ci_upper: torch.tensor, upper confidence interval for the corresponding targets
    """
    pred = pred.to(device)
    ci_lower = ci_lower.to(device)
    ci_upper = ci_upper.to(device)
    inside = (pred > ci_lower) & (pred < ci_upper)
    return inside.float().mean()

def compute_loss_mapgp(args, dataloader, predictions, lst_metrics=[], mll=None):
    """
    Computes loss metrics for MAPGaussianProcess predictions.

    :param args: arguments (must have model_type)
    :param dataloader: tuple (x_test, y_test)
    :param predictions: tensor (predicted means from model)
    :param lst_metrics: list of metrics to compute (e.g. ['rmse', 'mae', 'mll'])
    :param mll: marginal log likelihood (optional, used if 'mll' is in lst_metrics)
    :return: dictionary of computed loss metrics
    """
    losses = {}
    x_test, y_test = dataloader

    # Ensure everything is on the same device
    pred_mean = predictions.mean.view(y_test.shape).to(y_test.device)

    # Negative Marginal Log Likelihood
    if 'mll' in lst_metrics and mll is not None and args.model_type == 'megp':
        losses['nmll'] = torch.mean(-mll(predictions, y_test)).item()

    # Compute other metrics
    if 'rmse' in lst_metrics:
        losses['rmse'] = rmse(pred_mean, y_test)
    if 'mae' in lst_metrics:
        losses['mae'] = mae(pred_mean, y_test)
    if 'rae' in lst_metrics:
        losses['rae'] = rae(pred_mean, y_test)
    if 'rrse' in lst_metrics:
        losses['rrse'] = rrse(pred_mean, y_test)

    return losses


# def get_metrics(args, dataloader, predictions, metrics=[], mll=None):
#     predictions = predictions.to(device)
#     losses = {'nmll': -1}

#     is_batched = not isinstance(dataloader, tuple)

#     if is_batched:
#         nmll_list = []
#         y_batches = []
#         for idx, (_, y_batch) in enumerate(dataloader):
#             y_batch = y_batch.to(device)
#             y_batches.append(y_batch)
#             if 'mll' in metric_names and mll is not None and args.model_type not in ['ridge_reg', 'xgboost']:
#                 nmll_list.append((-mll(predictions[idx], y_batch)).mean().detach().cpu().item())
#         if nmll_list:
#             losses['nmll'] = np.mean(nmll_list)
#         test_y = torch.cat(y_batches, dim=0)
#     else:
#         test_x, test_y = dataloader
#         test_y = test_y.to(device)
#         if 'mll' in metric_names and mll is not None and args.model_type not in ['ridge_reg', 'xgboost']:
#             losses['nmll'] = (-mll(predictions, test_y)).mean().detach().cpu().item()

#     metric_fn_map = {
#         'rmse': rmse,
#         'mae': mae,
#         'rae': rea,
#         'rrse': rrse,
#     }

#     for metric in metric_names:
#         if metric in metric_fn_map:
#             losses[metric] = metric_fn_map[metric](predictions.view(test_y.shape), test_y).detach().cpu().numpy()

#     return losses


# In[16]:


# # ---------- Vectorized Kfull computation ----------

# class AbstractMeasurement(abc.ABC):
#     @abc.abstractmethod
#     def get_coordinate(self):
#         pass
    

# class AbstractPointMeasurement(AbstractMeasurement):
#     pass


# class PointMeasurement(AbstractMeasurement):
#     def __init__(self, coordinate):
#         self.coordinate = np.array(coordinate)

#     def get_coordinate(self):
#         return self.coordinate

    
# class dPointMeasurement(AbstractMeasurement):
#     def __init__(self, coordinate, derivative_index):
#         self.coordinate = np.array(coordinate)
#         self.derivative_index = derivative_index

#     def get_coordinate(self):
#         return self.coordinate

# class ddPointMeasurement(AbstractMeasurement):
#     def __init__(self, coordinate, derivative_indices):
#         self.coordinate = np.array(coordinate)
#         self.derivative_indices = derivative_indices # Tuple of (index1, index2) for the second derivative

#     def get_coordinate(self):
#         return self.coordinate
    
# class dddPointMeasurement(AbstractMeasurement):
#     def __init__(self, coordinate, derivative_indices):
#         self.coordinate = np.array(coordinate)
#         self.derivative_indices = derivative_indices # Tuple of (index1, index2, index3) for the third derivative

#     def get_coordinate(self):
#         return self.coordinate

# class ddddPointMeasurement(AbstractMeasurement):
#     def __init__(self, coordinate, derivative_indices):
#         self.coordinate = np.array(coordinate)
#         self.derivative_indices = derivative_indices # Tuple of (index1, index2, index3, index4) for the fourth derivative

#     def get_coordinate(self):
#         return self.coordinate

# class PointIndexMeasurement(AbstractMeasurement):
#     def __init__(self, index):
#         self.index = index

#     def get_coordinate(self):
#         return self.index
    
# # below are the measurements used for PDE solving    
# class ΔδPointMeasurement(AbstractPointMeasurement):
#     def __init__(self, coordinate, weight_Δ=1.0, weight_δ=0.0):
#         self.coordinate = np.array(coordinate)
#         self.weight_Δ = weight_Δ
#         self.weight_δ = weight_δ
        
#     def get_coordinate(self):
#         return self.coordinate
    
# class ΔΔδPointMeasurement(AbstractPointMeasurement):
#     def __init__(self, coordinate, weight_ΔΔ, weight_Δ=1.0, weight_δ=0.0):
#         self.coordinate = np.array(coordinate)
#         self.weight_ΔΔ = np.array(weight_ΔΔ)
#         self.weight_Δ = weight_Δ
#         self.weight_δ = weight_δ
#     def get_coordinate(self):
#         return self.coordinate
    
# class ΔΔΔPointMeasurement(AbstractPointMeasurement):
#     def __init__(self, coordinate, weight_Δ11=0.0, weight_Δ12=0.0, weight_Δ22=0.0):
#         self.coordinate = np.array(coordinate)
#         self.weight_Δ11 = weight_Δ11
#         self.weight_Δ12 = weight_Δ12
#         self.weight_Δ22 = weight_Δ22
#     def get_coordinate(self):
#         return self.coordinate

# def _jacobian(f, x, h=1e-8):
#     """Computes the Jacobian of f at x using central differences."""
#     x = np.asarray(x)  # Ensure x is an array
#     fx = np.asarray(f(x))  # Ensure fx is an array
#     n = len(x)
#     m = fx.size # change here, use size instead of len
#     J = np.zeros((m, n))
#     for i in range(n):
#         x_plus_h = x.copy()
#         x_plus_h[i] += h
#         J[:, i] = (np.asarray(f(x_plus_h)) - fx).reshape(m) / h   # Make f(x_plus_h) an array, and reshape
#     return J

# def _hessian(f, x, h=1e-5):
#     """Computes the Hessian of f at x using central differences."""
#     x = np.asarray(x)
#     fx = np.asarray(f(x))
#     n = len(x)
#     H = np.zeros((n, n))
#     for i in range(n):
#         x_plus_h = x.copy()
#         x_plus_h[i] += h
#         H[i, :] = (_jacobian(f, x_plus_h, h) - _jacobian(f, x, h)) / h
#     return (H + H.T) / 2

# class AbstractCovarianceFunction(abc.ABC):
#     def _init_(self, device=None):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
        
#     @abc.abstractmethod
#     def __call__(self, x, y):
#         pass

#     def batched_call(self, out, x_vec, y_vec):
#         for i in range(out.shape[0]):
#             for j in range(out.shape[1]):
#                 out[i, j] = self(x_vec[i], y_vec[j])

#     def batched_call_symmetric(self, out, x_vec):
#         for i in range(out.shape[0]):
#             for j in range(i, out.shape[1]):
#                 out[i, j] = self(x_vec[j], x_vec[i])
#                 if i != j:
#                     out[j, i] = out[i, j]
                    
# class GaussianCovariance_generic(AbstractCovarianceFunction):
#     def __init__(self):
#         self.lengthscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))
#         self.outputscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))
        
#         self.lengthscale = float(lengthscale)
#         self.outputscale = float(outputscale)

#         # device & dtype
#         self.device = torch.device("cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu"))
#         self.dtype = dtype or torch.get_default_dtype()

#         # constants
#         self.a = float(torch.sqrt(torch.tensor(5.0, dtype=self.dtype)).item())  
#         self.eps = 1e-12  
        
#     def matern52(self, diff, lengthscale):
#         diff = diff.to(device=self.device, dtype=self.dtype)
#         ls = torch.tensor(lengthscale, device=self.device, dtype=self.dtype)

#         s = torch.sqrt(torch.clamp((diff**2).sum(), min=0.0))  
#         r = s / (ls + 0.0)
#         sqrt5_r = self.a * r
#         factor = 1.0 + sqrt5_r + (5.0/3.0) * (r**2)
#         k = factor * torch.exp(-sqrt5_r)
#         return k, r, s

#     def __call__(self, a_inp: AbstractMeasurement, b_inp: AbstractMeasurement):
#         x = torch.as_tensor(a_inp.get_coordinate(), device=self.device, dtype=self.dtype)
#         y = torch.as_tensor(b_inp.get_coordinate(), device=self.device, dtype=self.dtype)

#         diff = x - y                      
#         D = diff.numel()
#         ls = self.lengthscale
#         out_scale = self.outputscale

#         k_scalar, r, s = self.matern52(diff, ls)  
        
#         exp_term = torch.exp(-self.a * r)

    
#         is_small = (r < self.eps)

#         # point-point 
#         if isinstance(a_inp, PointMeasurement) and isinstance(b_inp, PointMeasurement):
#             return torch.as_tensor(out_scale, device=self.device, dtype=self.dtype) * k_scalar

#         # point-grad 
#         elif isinstance(a_inp, PointMeasurement) and isinstance(b_inp, dPointMeasurement):
#             j = b_inp.derivative_index
#             if is_small:
#                 return torch.as_tensor(0.0, device=self.device, dtype=self.dtype)
#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)
#             val = - fprime * (diff[j] / (ls * s))
#             return out_scale * val
        
        
#         # point-hessian 
#         elif isinstance(a_inp, PointMeasurement) and isinstance(b_inp, ddPointMeasurement):
#             j1, j2 = b_inp.derivative_indices
#             if is_small:
#                 f2_0 = -5.0/3.0
#                 delta = 1.0 if j1 == j2 else 0.0
#                 val = f2_0 * delta / (ls**2)
#                 return out_scale * torch.as_tensor(val, device=self.device, dtype=self.dtype)

#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)
#             f2 = (5.0/3.0) * (5.0 * (r**2) - self.a * r - 1.0) * torch.exp(-self.a * r)

#             dr_y_b = - diff[j2] / (ls * s)
#             dr_y_a = - diff[j1] / (ls * s)
#             second_r = (- (1.0 if j1 == j2 else 0.0) / (ls * s) + (diff[j1] * diff[j2]) / (ls * s**3))

#             val = f2 * (dr_y_a * dr_y_b) + fprime * second_r
#             return out_scale * val

#         # grad-point 
#         elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, PointMeasurement):
#             i = a_inp.derivative_index
#             if is_small:
#                 return torch.as_tensor(0.0, device=self.device, dtype=self.dtype)
#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)
#             val = fprime * (diff[i] / (ls * s))
#             return out_scale * val

#         #  grad-grad 
#         elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, dPointMeasurement):
#             i = a_inp.derivative_index
#             j = b_inp.derivative_index

#             if is_small:
#                 diag = (5.0/3.0) * (1.0 / (ls**2))
#                 return out_scale * (torch.as_tensor(diag if i == j else 0.0, device=self.device, dtype=self.dtype))
#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)           
#             f2 = (5.0/3.0) * (5.0 * (r**2) - self.a * r - 1.0) * torch.exp(-self.a * r)         

#             d_i = diff[i]
#             d_j = diff[j]
#             term1 = - f2 * (d_i * d_j) / (ls**2 * s**2)
#             term2 = - fprime * ( (1.0 if i == j else 0.0) / (ls * s) - (d_i * d_j) / (ls * s**3) )
#             val = term1 + term2
#             return out_scale * val
        
#        # grad-hessian 
#         elif isinstance(a_inp, dPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
#             i = a_inp.derivative_index
#             j1, j2 = b_inp.derivative_indices
#             if is_small:
#                 return torch.as_tensor(0.0, device=self.device, dtype=self.dtype)
#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)
#             f2 = (5.0/3.0) * (5.0*(r**2) - self.a*r - 1.0) * torch.exp(-self.a*r)
#             term = ( (1.0 if i==j1 else 0.0) * diff[j2] + 
#                      (1.0 if i==j2 else 0.0) * diff[j1] +
#                      f2 * diff[i]*diff[j1]*diff[j2]/(ls**3*s**3) ) / (ls**3)
#             return out_scale * term


#         # hessian-point 
#         elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, PointMeasurement):
#             i1, i2 = a_inp.derivative_indices
#             if is_small:
#                 f2_0 = -5.0/3.0
#                 delta = 1.0 if i1 == i2 else 0.0
#                 val = f2_0 * delta / (ls**2)
#                 return out_scale * torch.as_tensor(val, device=self.device, dtype=self.dtype)

#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)
#             f2 = (5.0/3.0) * (5.0 * (r**2) - self.a * r - 1.0) * torch.exp(-self.a * r)

#             dr_x_i1 = diff[i1] / (ls * s)
#             dr_x_i2 = diff[i2] / (ls * s)
#             second_r_x = ( (1.0 if i1 == i2 else 0.0) / (ls * s) - (diff[i1] * diff[i2]) / (ls * s**3) )

#             val = f2 * (dr_x_i1 * dr_x_i2) + fprime * second_r_x
#             return out_scale * val
        
#         # hessian-grad 
#         elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, dPointMeasurement):
#             i1, i2 = a_inp.derivative_indices
#             j = b_inp.derivative_index

#             if is_small:
#                 return torch.as_tensor(0.0, device=self.device, dtype=self.dtype)

#             fprime = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)
#             f2 = (5.0/3.0) * (5.0*(r**2) - self.a*r - 1.0) * torch.exp(-self.a*r)

#             dr_j = - diff[j] / (ls * s)
#             dr_i1 = diff[i1] / (ls * s)
#             dr_i2 = diff[i2] / (ls * s)

#             d2r_i1i2 = (1.0 if i1==i2 else 0.0)/(ls*s) - (diff[i1]*diff[i2])/(ls*s**3)

#             val = f2 * dr_i1 * dr_i2 * dr_j + fprime * d2r_i1i2 * dr_j
#             return out_scale * val

#         #  hessian-hessian 
#         elif isinstance(a_inp, ddPointMeasurement) and isinstance(b_inp, ddPointMeasurement):
#             i1, i2 = a_inp.derivative_indices
#             j1, j2 = b_inp.derivative_indices

#             if is_small:
#                 delta_ab = (1.0 if i1 == i2 else 0.0)
#                 delta_cd = (1.0 if j1 == j2 else 0.0)
#                 val = (25.0/9.0) * (delta_ab * delta_cd) / (ls**4)  
#                 return out_scale * torch.as_tensor(val, device=self.device, dtype=self.dtype)

#             f1 = -(5.0/3.0) * r * (1.0 + self.a * r) * torch.exp(-self.a * r)        
#             f2 = (5.0/3.0) * (5.0 * (r**2) - self.a * r - 1.0) * torch.exp(-self.a * r) 
#             f3 = (25.0/3.0) * r * (3.0 - self.a * r) * torch.exp(-self.a * r)         
#             f4 = (25.0/3.0) * (5.0 * (r**2) - 5.0 * self.a * r + 3.0) * torch.exp(-self.a * r) 

#             d_i = diff[i1]; d_j = diff[i2]; d_k = diff[j1]; d_l = diff[j2]
#             delta_ij = 1.0 if i1 == i2 else 0.0
#             delta_kl = 1.0 if j1 == j2 else 0.0

#             term_ffff = f4 * (d_i/(ls*s)) * (d_j/(ls*s)) * (-d_k/(ls*s)) * (-d_l/(ls*s))
#             geom_term = (
#                 f3 * ( (d_i/(ls*s)) * ( - ( (1.0 if j1==j2 else 0.0)/(ls*s) - (d_k*d_l)/(ls*s**3) ) )
#                        + (d_j/(ls*s)) * ( - ( (1.0 if j1==j2 else 0.0)/(ls*s) - (d_k*d_l)/(ls*s**3) ) ) )
#                 + f2 * ( ( (1.0 if i1==i2 else 0.0)/(ls*s) - (d_i*d_j)/(ls*s**3) ) * ( (1.0 if j1==j2 else 0.0)/(ls*s) - (d_k*d_l)/(ls*s**3) ) )
#             )
#             val = term_ffff + geom_term

#             return out_scale * val

#         else:
#             raise NotImplementedError("Measurement type combination not supported for Matérn-5/2 (up to 2nd order).")

# # compute the new Kfull now
# # model = get_model(args, data, likelihood)
# # X_train = model.train_x.detach().cpu().numpy().T
# X_train = data.train_trans.x.detach.cpu().numpy().T

# def get_measurements(X_dom):
#     d, N = X_dom.shape
#     measurements = []
#     measurements.extend([PointMeasurement(X_dom[:, i]) for i in range(N)])
#     measurements.extend([dPointMeasurement(X_dom[:, i], (j)) for i in range(N) for j in range(d)])
#     measurements.extend([ddPointMeasurement(X_dom[:, i], (j, j)) for i in range(N) for j in range(d)])
    
#     return measurements

# def build_Kfull(cov, measurements):
#     N = len(measurements)
#     K_full = np.zeros(N, N)
    
#     for i in range(N):
#         for j in range(i, N):
#             value = cov(measurements[i], measurements[j])
#             K_full[i, j] = value.item() if isinstance(value, torch.Tensor) else value
#             K_full[j, i] = K_full[i, j]
            
#     return K_full

# measurements = get_measurements(X_train)
# cov = GaussianCovariance_generic()

# K_full = build_Kfull(cov_func, measurements)
# print("K_full shape:", K_full.shape)

# # use this new Kfull to get the new acq function
# # factorize the new kfll-clustring sparse cholesky factorization with dynamic supernode rho
# # run on the computational cost and get the data for the computational cost


# # **MAP ESTIMATE GAUSSIAN PROCESS REGRESSION (MEGP)**

# In[17]:


# class Model(abc.ABC):
#     @abc.abstractmethod
#     def forward(self, x):
#         pass

#     @abc.abstractmethod
#     def prediction(self, args, dataloader):
#         pass

#     @abc.abstractmethod
#     def loss(self):
#         pass

#     @abc.abstractmethod
#     def fit(self, train_data, args, debug):
#         pass
    
# class FixedCovarianceKernel(gpytorch.kernels.Kernel):
#     has_lengthscale = False

#     def __init__(self, K_full: torch.Tensor):
#         super().__init__()
#         self.register_buffer("K_full", K_full)

#     def forward(self, x1, x2=None, **params):
#         if x2 is None:
#             x2 = x1
#         i = x1.long().squeeze(-1)
#         j = x2.long().squeeze(-1)
#         return self.K_full[i][:, j]

# class MAPGaussianProcess(Model, gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood, mean_type="zero", covar_type=None, ard=True, args=None):
#         super().__init__(train_x, train_y, likelihood)
#         self.mean_module = self._mean(mean_type)
#         self.covar_module = self._covar(covar_type, ard)
#         self.train_x = train_x
#         self.train_y = train_y
#         self.args = args
#         self.to(device)
        
#         trunc_indices = torch.tensor([0, 1, 2]) # only need the top left hand part (3x3) for functions only
#         self.kernel_full = FixedCovarianceKernel(K_full)
#         K_trunc = K_full[trunc_indices][:, trunc_indices]
#         self.kernel_trunc = FixedCovarianceKernel(K_trunc)
        
#         self.train_indices = torch.tensor(9)

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#     def prediction(self, args, x_input):
#         self.eval()
#         self.likelihood.eval()

#         if not isinstance(x_input, torch.Tensor):
#             raise TypeError("Expected x_input to be a torch.Tensor")

#         if x_input.dim() != 2 or x_input.size(1) != 1:
#             raise ValueError("x_input must be index-based with shape (N, 1)")
    
#         if not torch.is_floating_point(x_input):
#             x_input = x_input.long()

#         if x_input.min() < 0 or x_input.max() >= self.covar_module.K_full.size(0):
#             raise IndexError("x_input indices out of bounds")
            
#         x_input = x_input.to(device)

#         with torch.no_grad(), \
#              gpytorch.settings.max_cholesky_size(10000), \
#              gpytorch.settings.cg_tolerance(0.01), \
#              gpytorch.settings.fast_pred_var():
#             latent_dist = self(x_input)
#             observed_dist = self.likelihood(latent_dist)

#         return {
#             "predictions": observed_dist,
#             "mean": observed_dist.mean.detach(),
#             "stddev": observed_dist.stddev.detach()
#         }


#     def loss(self):
#         return gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

#     def fit(self, train_data, args, debug):
#         x, y = train_data
#         self.train()
#         self.likelihood.train()

#         optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
#         loss_fn = self.loss()

#         epochs = args.n_epochs if args else 1000
#         runs = args.n_runs if args else 1

#         best_loss = float("inf")
#         best_losses = []

#         x, y = x.to(device), y.to(device)

#         for run in range(runs):
#             losses = []
#             for epoch in range(epochs):
#                 optimizer.zero_grad()
#                 with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.cg_tolerance(0.01):
#                     output = self(x)
#                     loss = -loss_fn(output, y)
#                     loss.backward()
#                 optimizer.step()
#                 scheduler.step(loss)
#                 losses.append(loss.item())

#                 if debug:
#                     print(f"[Run {run+1}, Epoch {epoch+1}] Loss: {loss.item():.6f}")

#                 if epoch > 15 and np.allclose(losses[-15:], losses[-1], rtol=0, atol=1e-4):
#                     break

#                 if losses[-1] < best_loss:
#                     best_loss = losses[-1]
#                     best_losses = list(losses)

#         return best_losses[-1], best_losses
    
#     @property
#     def batch_model(self):
#         return self
    
#     def _mean(self, mean_type: str):
#         if mean_type == "zero":
#             return gpytorch.means.ZeroMean()
#         elif mean_type == "constant":
#             return gpytorch.means.ConstantMean()
#         else:
#             raise ValueError(f"Unsupported mean type: {mean_type}")

#     def _covar(self, kernel: str, ard: bool):
#         if kernel == "kernel_full":
#             return self.kernel_full
#         elif kernel == "kernel_trunc":
#             return self.kernel_trunc 
#         else:
#             raise ValueError(f"Unsupported kernel type: {kernel}")
            
# #         ard_num_dims = self.train_inputs[0].shape[1] if ard else None

# #         lengthscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))
# #         outputscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))

# #         if kernel == "RBF":
# #             base_kernel = gpytorch.kernels.RBFKernel(
# #                 ard_num_dims=ard_num_dims,
# #                 lengthscale_prior=lengthscale_prior
# #             )
# # #         if kernel == "Matern52":
# # #             base_kernel = gpytorch.kernels.MaternKernel(
# # #                 nu=2.5,
# # #                 ard_num_dims=ard_num_dims,
# # #                 lengthscale_prior=lengthscale_prior
# # #             )
# #         else:
# #             raise ValueError(f"Unsupported kernel type: {kernel}")

# #         return gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
    
    
# def get_model(args, data, likelihood):
#     """
#     Returns the MAP GP model (MAPGaussianProcess) with appropriate mean and covariance configuration.

#     :param args: Arguments object (must have `mean_type`, `covar_type`, `ard` attributes)
#     :param data: An object containing training data (expects `data.train_trans.x`, `data.train_trans.y`)
#     :param likelihood: A GPyTorch likelihood object, e.g. gpytorch.likelihoods.GaussianLikelihood
#     """
# #     train_x = data.train_trans.x
#     train_x = torch.arange(N, device=K_full.device).unsqueeze(-1)
#     train_y = data.train_trans.y

#     model = MAPGaussianProcess(
#         train_x=train_x,
#         train_y=train_y,
#         likelihood=likelihood,
#         mean_type=getattr(args, "mean_type", "zero"),
#         covar_type = args.covar_type if args.covar_type is not None else args.kernel_type,
#         ard=getattr(args, "ard", True),
#         args = args
#     )

#     return model


# # **ACQUISITION FUNCTIONS**

# In[18]:


# class FixedCovarianceKernel(gpytorch.kernels.Kernel):
#     has_lengthscale = False

#     def __init__(self, K_full: torch.Tensor):
#         super().__init__()
#         self.register_buffer("K_full", K_full)

#     def forward(self, x1, x2=None, **params):
#         if x2 is None:
#             x2 = x1
#         i = x1.long().squeeze(-1)
#         j = x2.long().squeeze(-1)
#         return self.K_full[i][:, j]

# For the MI acq function, I need to define the following two things for the code to run:
# model.train_indices  # indices of observed measurements-these will be (0 to 8) x (0 to 8)
# trunc_indices        # indices defining truncated subspace-these will be (0 to 2) x (0 to 2)-done
# also, I need to replace the K_full part with the Cholesky factorization

class BaseAcquisitionFunction(abc.ABC):
    def __init__(self, model):
#         trunc_indices = torch.tensor([0, 1, 2]) # only need the top left hand part (3x3) for functions only
        self.model = model
        self.device = model.device if hasattr(model, "device") else next(model.parameters()).device
#         self.kernel_full = FixedCovarianceKernel(K_full)
#         K_trunc = K_full[trunc_indices][:, trunc_indices]
#         self.kernel_trunc = FixedCovarianceKernel(K_trunc)
        #self.candidate_points = candidate_points.to(self.device)

    @abc.abstractmethod
    def evaluate(self, x):
        pass 
    
#     def _build_kernel_trunc(self, ard):   
#         ard_num_dims = self.model.train_inputs[0].shape[1] if ard else None

#         lengthscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))
#         outputscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))

#         base_kernel = gpytorch.kernels.RBFKernel(
#             ard_num_dims=ard_num_dims,
#             lengthscale_prior=lengthscale_prior
#         )

#         return gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)

class BaseBayesianAcquisitionFunction(BaseAcquisitionFunction):
    def __init__(self, model):
        super().__init__(model)
        if self.model.batch_model is None:
            model.set_batch_model()
        self.model.batch_model.eval()

    @abc.abstractmethod
    def evaluate(self, x):
        pass

class VarianceAcquisition(BaseBayesianAcquisitionFunction):
    def evaluate(self, x):
        x = x.to(self.device)
        x = x.view(-1, 1) if x.ndim == 1 else x
        expanded_x = x.unsqueeze(0).repeat(
            self.model.args.num_samples * self.model.args.num_chains, 1, 1
        ).to(self.device)
        self.model.batch_model.eval()
        stddev = self.model.batch_model(expanded_x).stddev
        return torch.mean(stddev, dim=0)

class VarianceHessianAcquisition(BaseBayesianAcquisitionFunction):
    def __init__(self, model, lambd=0.5):
        super().__init__(model)
        self.lambd = lambd

    def compute_expected_trace_hessian(self, x):
        x = x.to(self.device).requires_grad_(True)
        num_samples = x.shape[0]
        traces = []

        for i in range(num_samples):
            xi = x[i].clone().detach().to(self.device).requires_grad_(True)
            output = self.model.batch_model(xi.unsqueeze(0)).mean
            grad = torch.autograd.grad(output, xi, create_graph=True)[0]

            hessian_rows = []
            for j in range(grad.shape[-1]):
                hess_j = torch.autograd.grad(grad[0, j], xi, retain_graph=True)[0]
                hessian_rows.append(hess_j)
            hess_matrix = torch.stack(hessian_rows, dim=0)
            trace = torch.trace(hess_matrix)
            traces.append(trace)

        expected_trace = torch.mean(torch.stack(traces)).to(self.device)
        return expected_trace.detach()

    def evaluate(self, x):
        x = x.to(self.device)
        x = x.view(-1, 1) if x.ndim == 1 else x

        expanded_x = x.unsqueeze(0).repeat(
            self.model.args.num_samples * self.model.args.num_chains, 1, 1
        ).to(self.device)

        output = self.model.batch_model(expanded_x)
        stddev = output.stddev
        var_term = torch.mean(stddev, dim=0)

        hess_term = []
        for i in range(x.shape[0]):
            xi = expanded_x[:, i:i+1, :]
            trace = self.compute_expected_trace_hessian(xi)
            hess_term.append(trace)

        hess_term = torch.stack(hess_term).to(self.device)
        return self.lambd * var_term + (1 - self.lambd) * hess_term

    
class ALM(BaseBayesianAcquisitionFunction):
    """
    Bayesian Active Learning MacKay (B-ALM).
    The Bayesian component comes from averaging over the models instead of using the mode.
    """
#     def evaluate(self, x):
#         x = x.to(self.device)
#         output = self.predictions
#         mean_variance = torch.mean(torch.pow(output.stddev, 2), dim=0)
#         return mean_variance
    
    
    def evaluate(self, x):
#         x = x.to(self.device)
#         x = x.view(-1, 1) if x.ndim == 1 else x
        
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.view(-1, 1)

        # Expand input for multiple samples
        n_samples = self.model.args.num_samples * self.model.args.num_chains
#         expanded_x = x.unsqueeze(0).repeat(n_samples, 1, 1)
        expanded_x = x.unsqueeze(0).expand(n_samples, *x.shape)

        self.model.batch_model.eval()
        output = self.model.batch_model(expanded_x)

        # Mean predictive variance across samples
        variance = output.stddev.pow(2)
        mean_variance = variance.mean(dim=0).squeeze(-1)  # shape: [N]
        return mean_variance
    
class BALD(BaseBayesianAcquisitionFunction):
#     def evaluate(self, x):
#         x = x.to(self.device)
#         x = x.view(-1, 1) if x.ndim == 1 else x
#         expanded_test_x = x.unsqueeze(0).repeat(
#             self.model.args.num_samples * self.model.args.num_chains, 1, 1
#         ).to(self.device)
#         self.model.batch_model.eval()
#         output = self.model.batch_model(expanded_test_x)

#         stddev = output.stddev
#         mean_stddev = torch.mean(stddev, dim=0)
#         entropy_expected = torch.log(mean_stddev + 1e-9)
#         expected_entropy = torch.mean(torch.log(stddev + 1e-9), dim=0)
#         return entropy_expected - expected_entropy
    
    def evaluate(self, x):
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.view(-1, 1)
           
        n_samples = self.model.args.num_samples * self.model.args.num_chains
        expanded_x = x.unsqueeze(0).expand(n_samples, *x.shape)
        
        self.model.batch_model.eval()
        output = self.model.batch_model(expanded_x)
        
        var = output.variance
        mean_var = var.mean(dim=0)
        
        entropy_expected = 0.5 * torch.log(mean_var + 1e-9)
        
        expected_entropy = 0.5 * torch.mean(torch.log(var + 1e-9), dim=0)

        return (entropy_expected - expected_entropy).squeeze(-1)

class H_BALD(BaseBayesianAcquisitionFunction):
    def __init__(self, model, beta=1.0):
        super().__init__(model)
        self.beta = beta

    def compute_hessian_variance_trace(self, x):
        x = x.to(self.device).requires_grad_(True)
        num_samples = x.shape[0]
        hessians = []

        for i in range(num_samples):
            xi = x[i].clone().detach().to(self.device).requires_grad_(True)
            output = self.model.batch_model(xi.unsqueeze(0)).mean
            grad = torch.autograd.grad(output, xi, create_graph=True)[0]
            hessian = []
            for j in range(grad.shape[-1]):
                hess_j = torch.autograd.grad(grad[0, j], xi, retain_graph=True)[0]
                hessian.append(hess_j)
            hess = torch.stack(hessian, dim=0)
            hessians.append(hess)

        hessians = torch.stack(hessians)
        var_hess = torch.var(hessians, dim=0)
        trace = torch.trace(var_hess)
        return trace.detach()

    def evaluate(self, x):
        x = x.to(self.device)
        x = x.view(-1, 1) if x.ndim == 1 else x
        expanded_test_x = x.unsqueeze(0).repeat(
            self.model.args.num_samples * self.model.args.num_chains, 1, 1
        ).to(self.device)
        self.model.batch_model.eval()
        output = self.model.batch_model(expanded_test_x)

        stddev = output.stddev
        mean_stddev = torch.mean(stddev, dim=0)
        entropy_expected = torch.log(mean_stddev + 1e-9)
        expected_entropy = torch.mean(torch.log(stddev + 1e-9), dim=0)
        bald = entropy_expected - expected_entropy

        hessian_term = []
        expanded_x = x.unsqueeze(0).repeat(
            self.model.args.num_samples * self.model.args.num_chains, 1, 1
        ).to(self.device)

        for i in range(x.shape[0]):
            xi_expanded = expanded_x[:, i:i+1, :]
            trace = self.compute_hessian_variance_trace(xi_expanded)
            hessian_term.append(trace)

        hessian_term = torch.stack(hessian_term).to(self.device)
        return bald + self.beta * hessian_term
  

    
class BQBC(BaseBayesianAcquisitionFunction):
    """
    Bayesian Query-by-Committee (B-QBC)
    """
#     def evaluate(self, x):
#         x = x.to(self.device)
#         x = x.view(-1, 1) if x.ndim == 1 else x
#         expanded_test_x = x.unsqueeze(0).repeat(
#             self.model.args.num_samples * self.model.args.num_chains, 1, 1
#         ).to(self.device)

#         self.model.batch_model.eval()
#         output = self.model.batch_model(expanded_test_x)

#         means = output.mean  
#         variance_of_means = torch.pow(torch.std(means.detach(), dim=0), 2)

#         return variance_of_means
    
    def evaluate(self, x):
        x = x.to(self.device, dtype=torch.float32)
        
        if x.ndim == 1:
            x = x.view(-1, 1)
         
        n_samples = self.model.args.num_samples * self.model.args.num_chains
        expanded_x = x.unsqueeze(0).expand(n_samples, *x.shape)
        
        self.model.batch_model.eval()
        
        output = self.model.batch_model(expanded_x)
        
        means = output.mean
        
        return means.var(dim=0, unbiased=False).squeeze(-1) 

class BALM(BaseBayesianAcquisitionFunction):
    """
    Bayesian Active Learning MacKay (B-ALM).
    Computes mean variance across model samples without using precomputed predictions.
    """
#     def evaluate(self, x):
#         # Move input to the correct device
#         x = x.to(self.device)
#         x = x.view(-1, 1) if x.ndim == 1 else x  # ensure 2D

#         # Expand input for multiple samples and chains
#         expanded_test_x = x.unsqueeze(0).repeat(
#             self.model.args.num_samples * self.model.args.num_chains, 1, 1
#         ).to(self.device)

#         # Set model to evaluation mode
#         self.model.batch_model.eval()
        
#         # Forward pass
#         output = self.model.batch_model(expanded_test_x)
#         stddev = output.stddev  # shape: [num_samples*num_chains, batch_size, output_dim]

#         # Compute mean variance across samples
#         mean_variance = torch.mean(stddev.pow(2), dim=0)

#         return mean_variance
    
    def evaluate(self, x):
        x = x.to(self.device, dtype=torch.float32)
        if x.ndim == 1:
            x = x.view(-1, 1)
            
        n_samples = self.model.args.num_samples * self.model.args.num_chains
        expanded_x = x.unsqueeze(0).expand(n_samples, *x.shape)
        
        self.model.batch_model.eval()
        
        output = self.model.batch_model(expanded_x)
        
        var = output.variance
        
        return var.mean(dim=0).squeeze(-1)
    
# class BQBCClassification(BaseBayesianAcquisitionFunction):
#     """
#     Bayesian Query-by-Committee for Classification.
#     Disagreement is measured using vote entropy across the committee.
#     """

#     def evaluate(self, x):
#         """
#         x: Tensor of shape [batch_size, input_dim]
#         Returns: Tensor of shape [batch_size] with vote entropy per point
#         """
#         if x is None:
#             # Handle gracefully: return a tensor of zeros (or very small value)
#             batch_size = 1  # default to 1 if no info about batch
#             return torch.zeros(batch_size, device=getattr(self, "device", "cpu"))

#         x = x.to(getattr(self, "device", "cpu"))  # fallback to cpu if device not set
#         x = x.view(-1, 1) if x.ndim == 1 else x  # ensure 2D

#         n_samples = self.model.args.num_samples * self.model.args.num_chains
#         expanded_x = x.unsqueeze(0).repeat(n_samples, 1, 1).to(getattr(self, "device", "cpu"))

#         self.model.batch_model.eval()
#         output = self.model.batch_model(expanded_x)

#         # Assuming output.mean returns class probabilities of shape [n_samples, batch_size, num_classes]
#         probs = output.mean  # shape: [n_samples, batch_size, num_classes]

#         # Compute average predicted probability per class across committee
#         avg_probs = probs.mean(dim=0)  # shape: [batch_size, num_classes]

#         # Compute vote entropy per point
#         vote_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9), dim=-1)  # shape: [batch_size]

#         return vote_entropy

class BQBCClassification(BaseBayesianAcquisitionFunction):
    """
    Bayesian Query-by-Committee for Classification (index-based).

    Disagreement is measured using vote entropy across the committee
    for candidate measurement indices.
    """

    def evaluate(self, x):
        # x: candidate measurement indices
        if x is None:
            batch_size = 1
            return torch.zeros(batch_size, device=getattr(self, "device", "cpu"))

        # ensure long and on correct device
        x = x.long().to(self.device)

        # ensure shape [N, 1]
        if x.ndim == 1:
            x = x.view(-1, 1)

        # number of posterior samples / committee members
        n_samples = self.model.args.num_samples * self.model.args.num_chains

        # expand indices across samples
        expanded_x = x.unsqueeze(0).expand(n_samples, *x.shape)

        # set model to eval mode
        self.model.batch_model.eval()

        # forward pass: predictive distribution
        output = self.model.batch_model(expanded_x)

        # predicted class probabilities: [n_samples, N, num_classes]
        probs = output.mean

        # average predicted probability per class across committee
        avg_probs = probs.mean(dim=0)  # shape: [N, num_classes]

        # vote entropy per measurement index
        vote_entropy = -torch.sum(avg_probs * torch.log(avg_probs + 1e-9), dim=-1)  # shape: [N]

        return vote_entropy

# also, I need to replace the K_full part with the Cholesky factorization

# check this function as well
# class MutualInfoAcquisition(BaseBayesianAcquisitionFunction):
#     """
#     Posterior kernel mutual information between
#     full kernel GP and truncated kernel GP.

#     Measures remaining function-space uncertainty reduction
#     due to kernel enrichment (e.g. derivatives).
#     """

#     def evaluate(self, x):
#         # Move inputs to device
# #         x = x.to(self.device)
# #         x = x.view(-1, 1) if x.ndim == 1 else x
# #         X_train = self.train.x.to(self.device)
        
#         x = x.long().to(self.device).view(-1, 1)
#         X_train = self.model.train_indices.long().to(self.device).view(-1, 1)
    
#         # number of points
#         x_number = x.shape[0]
#         X_train_number = X_train.shape[0]

#         # ---- FULL KERNEL ----
#         K_xx_f = self.model.kernel_full(x, x)
#         K_xX_f = self.model.kernel_full(x, X_train)
#         K_XX_f = self.model.kernel_full(X_train, X_train)

#         # ---- TRUNCATED KERNEL ---
#         K_xx_t = self.model.kernel_trunc(x, x)
#         K_xX_t = self.model.kernel_trunc(x, X_train)
#         K_XX_t = self.model.kernel_trunc(X_train, X_train)

#         # Jitter for numerical stability
#         jitter_x = 1e-6 * torch.eye(x_number, device=self.device)
#         jitter_X = 1e-6 * torch.eye(X_train_number, device=self.device)

#         K_XX_f = K_XX_f + jitter_X
#         K_XX_t = K_XX_t + jitter_X
        
#         K_xx_f = K_xx_f + jitter_x
#         K_xx_t = K_xx_t + jitter_x


#         # Cholesky factorizations
#         L_f = torch.linalg.cholesky(K_XX_f)
#         L_t = torch.linalg.cholesky(K_XX_t)

#         # Solve K_XX^{-1} K_Xx efficiently
#         # full kernel
#         A_f = torch.cholesky_solve(K_xX_f.T, L_f)
#         K_cond_f = K_xx_f - K_xX_f @ A_f
#         K_cond_f = K_cond_f + jitter_x

#         # truncated kernel
#         A_t = torch.cholesky_solve(K_xX_t.T, L_t)
#         K_cond_t = K_xx_t - K_xX_t @ A_t
#         K_cond_t = K_cond_t + jitter_x

#         # Cholesky of conditional kernels
#         L_cf = torch.linalg.cholesky(K_cond_f)
#         L_ct = torch.linalg.cholesky(K_cond_t)

#         # log-determinants
#         logdet_f = 2.0 * torch.sum(torch.log(torch.diag(L_cf)))
#         logdet_t = 2.0 * torch.sum(torch.log(torch.diag(L_ct)))

#         # Mutual Information
#         mutual_info = 0.5 * (logdet_f - logdet_t)
#         mutual_info = torch.clamp(mutual_info, min=0.0)


#         return mutual_info
    
# class MutualInfoKfullAcquisition(BaseBayesianAcquisitionFunction):
#     """
#     Mutual Information acquisition function without a candidate pool.
#     Evaluates MI for a given batch `x`, using the model's training data internally.
#     Mimics ALM's input handling and expansion.
#     """

#     def evaluate(self, x):
#         """
#         Evaluate mutual information for a batch of points `x`.
#         x: Tensor of shape [batch_size, input_dim]
#         Returns: Tensor of shape [batch_size] with MI per point
#         """
#         if x is None:
#             raise ValueError("Input x cannot be None.")
#         x = x.to(self.device)
#         if x.ndim == 1:
#             x = x.view(1, -1)
#         B, d = x.shape

#         if self.X_train is None or self.y_train is None:
#             raise ValueError("Training data not found in the model.")
#         X_train, y_train = self.X_train, self.y_train

#         # Get kernel parameters from the model
# #         lengthscale, outputscale = _get_model_kernel_params(self.model)

#         # Number of stochastic samples (for DEBAL)
#         n_samples = getattr(self.model.args, "num_samples", 1) * getattr(self.model.args, "num_chains", 1)

#         scores = []
#         for i in range(B):
#             x_i = x[i:i+1]  # [1,D]

#             # Loop over stochastic samples
#             mi_samples = []
            
            
#             # HERE WHY DO YOU NEED THE MATERN52_KERNEL FUNCTIONM THAT I CREATED
#             for s in range(n_samples):
#                 var_plain = plain_gp_posterior_variance(X_train, y_train, x_i, matern52_kernel, lengthscale_prior, outputscale_prior)
#                 var_hess = hessian_augmented_posterior_variance(X_train, y_train, x_i, matern52_kernel, lengthscale, outputscale)
#                 var_joint = torch.sqrt(var_plain * var_hess + 1e-18)
#                 mi = 0.5 * torch.log((var_plain * var_hess + 1e-18) / (var_joint + 1e-18))
#                 mi_samples.append(mi)

#             # Average over stochastic samples
#             scores.append(torch.stack(mi_samples).mean())

#         return torch.tensor(scores, device=self.device)




class TopologyPersistenceDirect:
    """
    t(x) = persistence(x)

    persistence_x:
        Tensor of shape (N_candidates,)
        Stores persistence value for each local maxima
    """

    def __init__(self, persistence_x, device="cpu"):
        self.persistence_x = persistence_x.to(device)
        self.device = device

    def evaluate(self, x):
        x = torch.as_tensor(x, dtype=torch.long, device=self.device).reshape(-1)
        return self.persistence_x[x]
    
class TopologyPersistenceKernel:
    """
    t(x) = persistence(c) * exp(-||x-c||^2 / lambda^2)

    x_locations:
        (N_candidates, dim)

    c_locations:
        (N_candidates, dim)

    persistence_c:
        (N_candidates,)
    """

    def __init__(
        self,
        x_locations,
        c_locations,
        persistence_c,
        lambda_param=1.0,
        device="cpu"
    ):
        self.x_locations = x_locations.to(device)
        self.c_locations = c_locations.to(device)
        self.persistence_c = persistence_c.to(device)

        self.lambda_param = lambda_param
        self.device = device

    def evaluate(self, x):
        x = torch.as_tensor(x, dtype=torch.long, device=self.device).reshape(-1)

        x_pos = self.x_locations[x]
        c_pos = self.c_locations[x]

        diff = x_pos - c_pos

        dist_sq = torch.sum(diff**2, dim=1)

        kernel = torch.exp(
            -dist_sq / (self.lambda_param ** 2)
        )

        return self.persistence_c[x] * kernel
    
    
class CombinedAcquisitionOption1(BaseBayesianAcquisitionFunction):

    def __init__(
        self,
        model,
        persistence_x,
        x_locations,
        input_transform,
        alpha=0.5,
        device="cpu"
    ):
        super().__init__(model)

        self.bald = BALD(model)

        self.topology = TopologyPersistenceDirect(
            persistence_x,
            device=device
        )
        self.x_locations = x_locations.to(device)
        self.input_transform = input_transform
        self.topology_scale = torch.clamp(
            self.topology.persistence_x.max(), min=1e-9
        )
        self.alpha = alpha
        self.device = device

    def evaluate(self, x):
        raw_x = torch.as_tensor(x, dtype=torch.float32, device=self.x_locations.device)
        raw_x = raw_x.reshape(-1, self.x_locations.shape[1])
        nearest_maxima = find_nearest_maxima_index(raw_x, self.x_locations)
        b_val = self.bald.evaluate(self.input_transform(raw_x))
        t_val = self.topology.evaluate(nearest_maxima) / self.topology_scale
        t_val = t_val.to(b_val.device)

        return (
            self.alpha * b_val
            + (1 - self.alpha) * t_val
        )
    

class CombinedAcquisitionOption2(BaseBayesianAcquisitionFunction):
    def __init__(
        self,
        model,
        x_locations,
        c_locations,
        persistence_c,
        input_transform,
        lambda_param=1.0,
        alpha=0.5,
        device="cpu"
    ):
        super().__init__(model)

        self.bald = BALD(model)

        self.topology = TopologyPersistenceKernel(
            x_locations=x_locations,
            c_locations=c_locations,
            persistence_c=persistence_c,
            lambda_param=lambda_param,
            device=device
        )
        self.input_transform = input_transform
        self.topology_scale = torch.clamp(
            self.topology.persistence_c.max(), min=1e-9
        )
        self.alpha = alpha
        self.device = device

    def evaluate(self, x):
        if x is None:
            raise ValueError("x is None — optimizer failed")

        raw_x = torch.as_tensor(
            x, dtype=torch.float32, device=self.topology.x_locations.device
        ).reshape(-1, self.topology.x_locations.shape[1])
        nearest_maxima = find_nearest_maxima_index(
            raw_x, self.topology.x_locations
        )
        b_val = self.bald.evaluate(self.input_transform(raw_x))
        t_val = self.topology.evaluate(nearest_maxima) / self.topology_scale
        t_val = t_val.to(b_val.device)

        return (
            self.alpha * b_val
            + (1 - self.alpha) * t_val
        )


    
def get_acquisition_function(args, model, data=None):
    def model_coordinates(raw_x):
        if data is None:
            return raw_x
        transformed_x, _, _ = transform(
            raw_x,
            args.transformation_x,
            data.x_mean,
            data.x_stddev,
        )
        return transformed_x

    if args.selection_criteria == 'DEBAL':
        raise NotImplementedError(
            "DEBAL is not implemented in this version of the pipeline."
        )
    elif args.selection_criteria == 'BALD':
        return BALD(model)
    elif args.selection_criteria == 'ALM':
        return ALM(model)
#     elif args.selection_criteria == 'h-bald':
#         return H_BALD(model)
    elif args.selection_criteria == 'variance':
        return VarianceAcquisition(model)
    elif args.selection_criteria == 'variance-hessian':
        return VarianceHessianAcquisition(model)
    elif args.selection_criteria == 'bqbc':
        return BQBC(model)
    elif args.selection_criteria == 'balm':
        return BALM(model)
    elif args.selection_criteria == 'QBC':
        # This pipeline fits a regression GP, so QBC means disagreement of
        # regression committee means rather than classification vote entropy.
        return BQBC(model)
    
    elif args.selection_criteria == 'topology_acquisition_1':
        image_path = (
            getattr(args, "topology_image_path", None)
            or getattr(args, "image_path", None)
            or "mt_sugarloof_new_zealand.jpg"
        )
        topology = load_topology_data(
            image_path,
            getattr(args, "topology_persistence_path", "mt_persistence_pairs.csv"),
            top_k=getattr(args, "topology_top_k", 50),
        )
        if data is None or data.search_space.shape[1] != topology["x_locations"].shape[1]:
            raise ValueError(
                "topology_acquisition_1 requires an image simulator with the "
                "same two-dimensional coordinates as topology_image_path."
            )
        topology_device = next(model.parameters()).device
        return CombinedAcquisitionOption1(
            model,
            topology["persistence_x"],
            topology["x_locations"],
            input_transform=model_coordinates,
            alpha=0.5,
            device=topology_device,
        )
    elif args.selection_criteria == 'topology_acquisition_2':
        image_path = (
            getattr(args, "topology_image_path", None)
            or getattr(args, "image_path", None)
            or "mt_sugarloof_new_zealand.jpg"
        )
        topology = load_topology_data(
            image_path,
            getattr(args, "topology_persistence_path", "mt_persistence_pairs.csv"),
            top_k=getattr(args, "topology_top_k", 50),
        )
        if data is None or data.search_space.shape[1] != topology["x_locations"].shape[1]:
            raise ValueError(
                "topology_acquisition_2 requires an image simulator with the "
                "same two-dimensional coordinates as topology_image_path."
            )
        topology_device = next(model.parameters()).device
        return CombinedAcquisitionOption2(
            model,
            topology["x_locations"],
            topology["c_locations"],
            topology["persistence_c"],
            input_transform=model_coordinates,
            lambda_param=1.0,
            alpha=0.5,
            device=topology_device,
        )
    
#     elif args.selection_criteria == 'DEBAL':
#         return MutualInfoKfullAcquisition(model)
    else:
        raise NotImplementedError(f'Sorry, acquisition {args.selection_criteria} is not there for now.')


# # **ACTIVE LEARNING LOOP**

# In[19]:


def optimize_acquisition(acq_fn, bounds, init_x, n_restarts=10):
    best_val = -float("inf")
    best_x = None

    for _ in range(n_restarts):
        x = (bounds[0] + (bounds[1] - bounds[0]) * torch.rand_like(init_x)).clone().detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([x], max_iter=20)

        def closure():
            optimizer.zero_grad()
            acq_val = acq_fn.evaluate(x).sum()
            loss = -acq_val  # Maximize acquisition
            loss.backward()
            return loss

        try:
            optimizer.step(closure)
            val = acq_fn.evaluate(x).sum().item()
            if val > best_val:
                best_val = val
                best_x = x.detach().clone()
        except Exception as e:
            print("Optimization failed:", e)

    return best_x

def _legacy_get_oracle_label(args, new_points, data, k_samples=1, repeat_sampling=1, seed=None):
    #device = args.device if hasattr(args, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if new_points is None:
        print("[Warning] new_points is None. Skipping this iteration.")
        return None  # or return safe defaults if needed
    
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    query_points = new_points[0] if new_points.ndim == 3 else new_points
    query_points = query_points.reshape(-1, query_points.shape[-1])

    # Get oracle values (numpy)
    queried_labels = data.oracle.query(query_points)

    # Store query points in labeled pool (numpy, for tracking)
    data.pool_labeled = query_points

    # Repeat x and y values
    new_x = query_points[:k_samples].repeat(repeat_sampling, 1)
    new_y = queried_labels[:k_samples].repeat(repeat_sampling, 1)

    # Convert to GPU tensors
#     new_x_tensor = torch.tensor(new_x, dtype=torch.float32).to(device)
#     new_y_tensor = torch.tensor(new_y, dtype=torch.float32).to(device)
    
    if isinstance(new_x, np.ndarray):
        new_x_tensor = torch.from_numpy(new_x).float().to(device)
    else:
        new_x_tensor = new_x.float().to(device)

    if isinstance(new_y, np.ndarray):
        new_y_tensor = torch.from_numpy(new_y).float().to(device)
    else:
        new_y_tensor = new_y.float().to(device)
        
    if data.train.x is None:
        data.train.x = new_x_tensor
        data.train.y = new_y_tensor
    else:
        data.train.x = torch.cat([data.train.x.to(device), new_x_tensor], dim=0)

    # Make sure y tensors are 2D
    if data.train.y.ndim == 1:
        data.train.y = data.train.y.unsqueeze(-1)
    if new_y_tensor.ndim == 1:
        new_y_tensor = new_y_tensor.unsqueeze(-1)

    data.train.y = torch.cat([data.train.y.to(device), new_y_tensor], dim=0)


#     if data.train.x is None:
#         data.train.x = new_x_tensor
#         data.train.y = new_y_tensor
#     else:
#         data.train.x = torch.cat([data.train.x.to(device), new_x_tensor], dim=0)
#         data.train.y = torch.cat([data.train.y.to(device), new_y_tensor], dim=0)

    return data


def get_oracle_label(
    args,
    new_points,
    data,
    raw_indices,
    iteration,
    acquisition_scores,
    k_samples=1,
    repeat_sampling=1,
    seed=None,
):
    """Query selected raw rows and append them with their provenance.

    ``new_points`` must already be rows from ``data.search_space``.  Keeping
    this enforcement at the only mutation point prevents the exported active
    dataset from drifting away from the raw dataset it is meant to reference.
    """
    if new_points is None:
        raise RuntimeError("The acquisition optimizer did not return a point.")

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    query_points = torch.as_tensor(
        new_points, dtype=data.train.x.dtype, device=data.train.x.device
    ).reshape(-1, data.train.x.shape[1])
    raw_indices = torch.as_tensor(raw_indices, dtype=torch.long).reshape(-1)

    if query_points.shape[0] != k_samples or raw_indices.shape[0] != k_samples:
        raise ValueError("The number of selected points must equal k_samples.")

    queried_labels = _ensure_2d(data.oracle.query(query_points))
    data.record_observations(
        x=query_points,
        y=queried_labels,
        raw_indices=raw_indices,
        iteration=iteration,
        acquisition_scores=acquisition_scores,
        repeat_sampling=repeat_sampling,
    )
    return data


def repeated_experiments(data, repeat_sampling, new_points):
    y_all = []
    for _ in range(repeat_sampling):
        y_sample = data.oracle.query(new_points[0].reshape(-1, new_points.shape[1]))
        y_all = np.concatenate((np.array(y_all), y_sample))
    return y_all, new_points


def get_sorted_data_points(args, scores, search_space):
    if scores.dim() > 1:
        scores = torch.mean(scores, dim=1)

    descending_scores = torch.argsort(scores, descending=True)
    sorted_indices = descending_scores
    sorted_points = search_space[sorted_indices]
    return sorted_points, sorted_indices


def sampling(args, model, data, predictions):
    pass

# def active_learning(args, data, model, predictions):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     torch.set_num_threads(args.num_chains)
    

#     acq_fns = get_acquisition_function(args, model)

#     dim = data.train.x.shape[1]
#     # BOUNDS CAN BE PUT STATICALLY AS WELL USING THE FOLLOWING COMMENT (JUST UNCOMMENT AND CHANGE THE BOUNDS)
# #     bounds = torch.tensor([[-2.0] * dim, [2.0] * dim], dtype=torch.float32).to(device)
#     search_space = data.oracle.search_space().cpu()
#     bounds = torch.stack([torch.min(search_space, dim=0).values,
#                           torch.max(search_space, dim=0).values])
#     bounds = bounds.to(device)
#     init_x = torch.rand((1, dim), dtype=torch.float32).to(device)

#     selected_points = []
#     scores = []

#     for acq_fn in acq_fns:
#         selected_point = optimize_acquisition(acq_fn, bounds, init_x)
#         selected_points.append(selected_point)
#         score = acq_fn.evaluate(selected_point)
#         scores.append(score.item() if isinstance(score, torch.Tensor) else score)
#     max_score_idx = int(np.argmax(scores))
#     selected_point = selected_points[max_score_idx]
#     data = get_oracle_label(
#         args,
#         selected_point,
#         data,
#         k_samples=args.k_samples,
#         repeat_sampling=args.repeat_sampling,
#         seed=None,
#     )

#     torch.set_num_threads(1)
#     return None, selected_point, data



def _as_score_vector(scores, expected_length):
    """Convert scalar/batched acquisition values into one float per raw row."""
    if not torch.is_tensor(scores):
        scores = torch.as_tensor(scores)
    scores = scores.detach().reshape(-1).to("cpu", dtype=torch.float64)
    if scores.numel() == 1:
        scores = scores.repeat(expected_length)
    elif scores.numel() != expected_length:
        scores = scores.reshape(expected_length, -1).mean(dim=1)
    return scores.numpy()


def active_learning(args, data, model, predictions, iteration=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.num_chains)

    acq_fn = get_acquisition_function(args, model, data=data)

    dim = data.train.x.shape[1]
    is_topology_acquisition = args.selection_criteria.startswith("topology_acquisition_")
    if is_topology_acquisition:
        # Topology scores are defined in raw image-pixel coordinates.  The
        # combined acquisition transforms only its GP/BALD component.
        acquisition_search_space = data.search_space
    else:
        # Other acquisitions see the same transformed coordinates as the GP.
        acquisition_search_space, _, _ = transform(
            data.search_space,
            args.transformation_x,
            data.x_mean,
            data.x_stddev,
        )
    bounds = torch.stack([
        torch.min(acquisition_search_space, dim=0).values,
        torch.max(acquisition_search_space, dim=0).values,
    ]).to(device)
    init_x = torch.rand((1, dim), dtype=torch.float32).to(device)

    optimizer_point_acquisition = optimize_acquisition(acq_fn, bounds, init_x)
    if optimizer_point_acquisition is None:
        raise RuntimeError("The acquisition optimiser failed for every restart.")
    if is_topology_acquisition:
        optimizer_point = optimizer_point_acquisition
    else:
        optimizer_point, _, _ = transform(
            optimizer_point_acquisition,
            args.transformation_x,
            data.x_mean,
            data.x_stddev,
            inverse=True,
        )
    selected_points, raw_indices = data.select_available_pool_points(
        optimizer_point, k_samples=args.k_samples
    )
    if is_topology_acquisition:
        score = acq_fn.evaluate(selected_points)
    else:
        selected_points_transformed, _, _ = transform(
            selected_points,
            args.transformation_x,
            data.x_mean,
            data.x_stddev,
        )
        score = acq_fn.evaluate(selected_points_transformed)
    score_vector = _as_score_vector(score, expected_length=args.k_samples)

    if iteration is None:
        iteration = max(data.selection_steps, default=-1) + 1

    data = get_oracle_label(
        args,
        selected_points,
        data,
        raw_indices=raw_indices,
        iteration=iteration,
        acquisition_scores=score_vector,
        k_samples=args.k_samples,
        repeat_sampling=args.repeat_sampling,
        seed=None,
    )

    torch.set_num_threads(1)
    return score_vector, selected_points, data


# # **MAP ESTIMATE GAUSSIAN PROCESS REGRESSION (MEGP)**

# In[20]:


class Model(abc.ABC):
    @abc.abstractmethod
    def forward(self, x):
        pass

    @abc.abstractmethod
    def prediction(self, args, dataloader):
        pass

    @abc.abstractmethod
    def loss(self):
        pass

    @abc.abstractmethod
    def fit(self, train_data, args, debug):
        pass


class MAPGaussianProcess(Model, gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean_type="zero", covar_type="RBF", ard=True, args=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = self._mean(mean_type)
        self.covar_module = self._covar(covar_type, ard)
        self.train_x = train_x
        self.train_y = train_y
        self.args = args
        self.to(device)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


    def prediction(self, args, x_input):
        self.eval()
        self.likelihood.eval()

        if not isinstance(x_input, torch.Tensor):
            raise TypeError("Expected x_input to be a torch.Tensor")

        x_input = x_input.to(device)

        with torch.no_grad(),              gpytorch.settings.max_cholesky_size(10000),              gpytorch.settings.cg_tolerance(0.01),              gpytorch.settings.fast_pred_var():
            latent_dist = self(x_input)
            observed_dist = self.likelihood(latent_dist)

        return {
            "predictions": observed_dist,
            "mean": observed_dist.mean.detach(),
            "stddev": observed_dist.stddev.detach()
        }


    def loss(self):
        return gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

    def fit(self, train_data, args, debug):
        x, y = train_data
        self.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(
            self.parameters(), lr=(args.initial_lr if args else 0.1)
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        loss_fn = self.loss()

        epochs = args.n_epochs if args else 1000
        # ``n_runs`` belongs to the outer experiment loop.  Reusing it here
        # silently trained every model multiple times per AL iteration.
        runs = 1

        best_loss = float("inf")
        best_losses = []

        x, y = x.to(device), y.to(device)

        for run in range(runs):
            losses = []
            for epoch in range(epochs):
                optimizer.zero_grad()
                with gpytorch.settings.max_cholesky_size(10000), gpytorch.settings.cg_tolerance(0.01):
                    output = self(x)
                    loss = -loss_fn(output, y)
                    loss.backward()
                optimizer.step()
                scheduler.step(loss)
                losses.append(loss.item())

                if debug:
                    print(f"[Run {run+1}, Epoch {epoch+1}] Loss: {loss.item():.6f}")

                if epoch > 15 and np.allclose(losses[-15:], losses[-1], rtol=0, atol=1e-4):
                    break

                if losses[-1] < best_loss:
                    best_loss = losses[-1]
                    best_losses = list(losses)

        return best_losses[-1], best_losses
    
    @property
    def batch_model(self):
        return self
    
    def _mean(self, mean_type: str):
        if mean_type == "zero":
            return gpytorch.means.ZeroMean()
        elif mean_type == "constant":
            return gpytorch.means.ConstantMean()
        else:
            raise ValueError(f"Unsupported mean type: {mean_type}")

    def _covar(self, kernel: str, ard: bool):
        ard_num_dims = self.train_inputs[0].shape[1] if ard else None

        lengthscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))
        outputscale_prior = gpytorch.priors.LogNormalPrior(0.0, np.sqrt(3.0))

        if kernel == "RBF":
            base_kernel = gpytorch.kernels.RBFKernel(
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior
            )
        elif kernel == "Matern52":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
                ard_num_dims=ard_num_dims,
                lengthscale_prior=lengthscale_prior
            )
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

        return gpytorch.kernels.ScaleKernel(base_kernel, outputscale_prior=outputscale_prior)
    
    
def get_model(args, data, likelihood):
    """
    Returns the MAP GP model (MAPGaussianProcess) with appropriate mean and covariance configuration.

    :param args: Arguments object (must have `mean_type`, `covar_type`, `ard` attributes)
    :param data: An object containing training data (expects `data.train_trans.x`, `data.train_trans.y`)
    :param likelihood: A GPyTorch likelihood object, e.g. gpytorch.likelihoods.GaussianLikelihood
    """
    train_x = data.train_trans.x
    train_y = data.train_trans.y

    model = MAPGaussianProcess(
        train_x=train_x,
        train_y=train_y,
        likelihood=likelihood,
        mean_type=getattr(args, "mean_type", "zero"),
        covar_type=getattr(args, "covar_type", "RBF"),
        ard=getattr(args, "covar_ard", True),
        args = args
    )

    return model


# In[ ]:




