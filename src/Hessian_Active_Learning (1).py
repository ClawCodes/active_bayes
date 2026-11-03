#!/usr/bin/env python
# coding: utf-8

# # **IMPORT LIBRARIES**

# In[7]:


import torch
import numpy as np
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
from torch.utils.data import TensorDataset, DataLoader
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
        (1, dim) continuous point

    x_locations:
        (N_maxima, dim)

    Returns:
        index of nearest maxima
    """

    diff = x_locations - x_query

    dist_sq = torch.sum(
        diff ** 2,
        dim=1
    )

    idx = torch.argmin(dist_sq)

    return idx.view(1)

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


# Z already computed earlier
# ── Salomon function ──────────────────────────────────────────────────────────
def salomon(x, y):
    r = np.sqrt(x**2 + y**2)
    return 1 - np.cos(2 * np.pi * r) + 0.1 * r

n = 100
xmin, xmax = -10, 10
ymin, ymax = -10, 10

#salomon
x = np.linspace(xmin, xmax, n)
y = np.linspace(ymin, ymax, n)
X, Y = np.meshgrid(x, y)           # both shape (n, n)

# Z = salomon(X, Y)

#tiger
img       = Image.open("mt_sugarloof_new_zealand.jpg").convert("L")   # grayscale, uint8
Z         = np.array(img, dtype=np.float64)        # shape (height, width)
height, width = Z.shape
print(f"Image loaded: {width} x {height} px")

maxima_indices = find_local_maxima(Z)

birth, death, persistence = load_persistence_pairs(
    "mt_persistence_pairs.csv"
)

persistence_x = compute_persistence_x(
    Z,
    maxima_indices,
    birth,
    death
)

critical_locations, persistence_c_global =     compute_critical_locations(
        Z,
        birth,
        death,
        top_k=50
)

c_locations, persistence_c =     assign_nearest_critical(
        maxima_indices,
        critical_locations,
        persistence_c_global
)

print("Local maxima:", len(maxima_indices))
print("Critical points:", len(critical_locations))

x_locations = build_x_locations(
    maxima_indices,
    np.arange(width),
    np.arange(height)
)

print(x_locations.shape)
# (N_maxima, 2)


x_locations = torch.tensor(
    x_locations,
    dtype=torch.float32
)

c_locations = torch.tensor(
    c_locations,
    dtype=torch.float32
)

persistence_c = torch.tensor(
    persistence_c,
    dtype=torch.float32
)

persistence_x = torch.tensor(
    persistence_x,
    dtype=torch.float32
)


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
    def __init__(self, args, oracle):
        self.args = args
        self.oracle = oracle
        self.search_space = self.oracle.search_space()
        self.train = Dataset(*self.oracle.sample_initial_data(n_samples=self.args.initial_samples,
                                                    space_filling_design=self.args.space_filling_design))
        
        self.test = Dataset(*self.oracle.sample_initial_data(n_samples=self.args.test_samples,
                                                         space_filling_design="lhs"))

#         self.true_mean = torch.Tensor(self.oracle.mean(x=self.search_space)).to(device)
#         self.true_std = torch.Tensor(self.oracle.stddev(x=self.search_space)).to(device)
        
        self.true_mean = torch.Tensor(self.oracle.mean(x=self.search_space))
        self.true_std = torch.Tensor(self.oracle.stddev(x=self.search_space))
        self.pool_labeled = None
        self.oracle_labels = None
        self.true_mean_cp = None
        self.true_std_cp = None
#         self.candidate_points = None

               
        self.train_trans = Dataset(None, None)
        self.test_trans = Dataset(None, None)

        self.gp_train_loader_ori = None
        self.gp_test_loader_ori = None
        self.gp_train_loader = None
        self.gp_test_loader = None

        self.x_mean, self.x_stddev = None, None
        self.y_mean, self.y_stddev = None, None
        
    def transformed_data_loader(self):
        train_x_trans, _, _ = transform(self.train.x, self.args.transformation_x)
        train_y_trans, _, _ = transform(self.train.y, self.args.transformation_y)
        
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
    
def sampling_initial_points(n_samples, search_space):
    # https://scikit-optimize.github.io/stable/modules/generated/skopt.sampler.Lhs.html#skopt.sampler.Lhs
    if search_space.ndim == 1:
        bounds = [(np.min(search_space), np.max(search_space))]
    else:
        mins = np.min(search_space, axis=0)
        maxs = np.max(search_space, axis=0)
        bounds = [(low, high) for low, high in zip(mins, maxs)]

    # Create skopt space and sample using Latin Hypercube
    space = skopt.space.Space(bounds)
    lhs = skopt.sampler.Lhs(lhs_type="classic", criterion='maximin')
    x_sample = np.array(lhs.generate(space.dimensions, n_samples))

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
    if simulator == "gramacy1d":
        return GramacyAndLee1d(device=device)
    elif simulator == "higdon1d":
        return Higdon1d(device=device)
    elif simulator == 'rastrigin1d':
        return Rastrigin1d(device=device)
    elif simulator == "gramacy2d":
        return GramacyAndLee2d(device=device)
    elif simulator == "branin2d":
        return Branin2d(device=device)
    elif simulator == "ishigami3d":
        return Ishigami3d(device=device)
    elif simulator == "hartmann6d":
        return Hartmann6d(device=device)
    elif simulator == "sine1d":
        return Sine1d(device=device)
    elif simulator == "ackley6d":
        return Ackley6d(device=device)
    elif simulator == "salomon2D":
        return Salomon2D(device=device)
    elif simulator == "tiger":
        return TigerImage2D(device=device)
    elif simulator == "mountain":
        return MountainImage2D(device=device)
    else:
        raise NotImplementedError(f'Sorry, simulator {simulator} is not there for now.')


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
        
        x = x.long().to(self.device)
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
        x = x.long().to(self.device)
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
        x = x.long().to(self.device)
        
        if x.ndim == 1:
            x = x.view(-1, 1)
         
        n_samples = self.model.args.num_samples * self.model.args.num_chains
        expanded_x = x.unsqueeze(0).expand(n_samples, *x.shape)
        
        self.model.batch_model_eval()
        
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
        x = x.long().to(self.device)
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
        x = x.long().to(device)

        if x.ndim == 1:
            x = x.view(-1)

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

        x = x.long().to(device)

        if x.ndim == 1:
            x = x.view(-1)

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
        alpha=0.5,
        device="cpu"
    ):
        super().__init__(model)

        self.bald = BALD(model)

        self.topology = TopologyPersistenceDirect(
            persistence_x,
            device=device
        )

        self.alpha = alpha
        self.device = device

    def evaluate(self, x):

        b_val = self.bald.evaluate(x)

        t_val = self.topology.evaluate(x)
        t_val = t_val / (t_val.max() + 1e-9)

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

        self.alpha = alpha
        self.device = device

    def evaluate(self, x):

        if x is None:
            raise ValueError("x is None — optimizer failed")

        # convert coordinate → index
        idx = find_nearest_maxima_index(
            x,
            self.topology.x_locations
        )

        b_val = self.bald.evaluate(x)

        t_val = self.topology.evaluate(idx)

        t_max = torch.max(t_val)

        if t_max > 0:
            t_val = t_val / t_max

        return (
            self.alpha * b_val
            + (1 - self.alpha) * t_val
        )


    
def get_acquisition_function(args, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.selection_criteria == 'DEBAL':
        return MutualInfoAcquisition(model, device)
    elif args.selection_criteria == 'BALD':
        return BALD(model)
    elif args.selection_criteria == 'ALM':
        return ALM(model, device)
#     elif args.selection_criteria == 'h-bald':
#         return H_BALD(model)
    elif args.selection_criteria == 'variance':
        return VarianceAcquisition(model)
    elif args.selection_criteria == 'variance-hessian':
        return VarianceHessianAcquisition(model, device)
    elif args.selection_criteria == 'bqbc':
        return BQBC(model, device)
    elif args.selection_criteria == 'balm':
        return BALM(model, device)
    elif args.selection_criteria == 'QBC':
        return BQBCClassification(model)
    
    elif args.selection_criteria == 'topology_acquisition_1':
        return CombinedAcquisitionOption1(model, persistence_x, alpha=0.5, device='cpu')
    elif args.selection_criteria == 'topology_acquisition_2':
        return CombinedAcquisitionOption2(model, x_locations, c_locations, persistence_c, lambda_param=1.0, alpha=0.5, device='cpu')
    
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

def get_oracle_label(args, new_points, data, k_samples=1, repeat_sampling=1, seed=None):
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
    sorted_indices = index_descending(descending_scores)
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



def active_learning(args, data, model, predictions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(args.num_chains)

    acq_fn = get_acquisition_function(args, model)

    dim = data.train.x.shape[1]
    search_space = data.oracle.search_space().cpu()
    bounds = torch.stack([torch.min(search_space, dim=0).values,
                          torch.max(search_space, dim=0).values]).to(device)
    init_x = torch.rand((1, dim), dtype=torch.float32).to(device)

    selected_point = optimize_acquisition(acq_fn, bounds, init_x)
    score = acq_fn.evaluate(selected_point)

    data = get_oracle_label(
        args,
        selected_point,
        data,
        k_samples=args.k_samples,
        repeat_sampling=args.repeat_sampling,
        seed=None,
    )

    torch.set_num_threads(1)
    return None, selected_point, data


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

        optimizer = torch.optim.Adam(self.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        loss_fn = self.loss()

        epochs = args.n_epochs if args else 1000
        runs = args.n_runs if args else 1

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
        ard=getattr(args, "ard", True),
        args = args
    )

    return model


# In[ ]:




