"""Default configuration for ``Active_Learning_experiments_new.py``.

This is deliberately a valid Python module (not YAML saved with a ``.py``
suffix).  Run an experiment with:

    python Active_Learning_experiments_new.py --cfg config.py

or copy this file, edit ``DEFAULT_CONFIG``, and point ``--cfg`` at the copy.
"""

DEFAULT_CONFIG = {
    # Reproducibility
    "seed": 0,

    # Model
    "model_type": "megp",
    "outputs": 1,
    "noise_prior": True,
    "covar_prior": "lalprior",
    "covar_type": "RBF",
    "covar_ard": True,
    "mean_type": "zero",
    "kernel_type": "kernel_trunc",

    # Data / simulator.  sine1d makes this default runnable without external
    # image assets.  Set simulator='mountain' or 'tiger' when their image files
    # are available alongside the run.
    "al_type": "population_based",
    "simulator": "sine1d",
    "image_path": None,
    "noise_level": 0.05,
    "initial_samples": 3,
    "space_filling_design": "lhs",
    "test_samples": 1000,
    "transformation_x": "min_max_feature_scaling",
    "transformation_y": "standardize",

    # Active learning
    "selection_criteria": "variance",
    "acquisition_functions": ["variance"],
    "active_learning_steps": 60,
    "k_samples": 1,
    "repeat_sampling": 1,

    # Optimisation / inference
    "initial_lr": 0.1,
    "milestones": [300, 600],
    "n_epochs": 40,
    "n_runs": 1,
    "num_chains": 1,
    "num_samples": 1,
    "warmup_steps": 200,
    "predict_mcmc": "mode",

    # Result metrics and durable data products
    "folder": "outputs",
    "model_name": "sine1d",
    "datasets_dir": "datasets",
    "save_datasets": True,
    "save_every": 1,
    "write_csv": True,
    # NPZ is always written.  CSV is omitted only for raw sets above this
    # threshold, preventing enormous duplicated text files for 6-D pools.
    "max_csv_rows": 250_000,
    "raw_export_chunk_size": 65_536,
    "export_test_data": True,

    # Needed only by topology_acquisition_1 / topology_acquisition_2.
    # Leave this None to use image_path for a mountain/tiger topology run.
    "topology_image_path": None,
    "topology_persistence_path": "mt_persistence_pairs.csv",
    "topology_top_k": 50,
}
