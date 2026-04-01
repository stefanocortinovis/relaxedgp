# Inverse-Free Sparse Variational Gaussian Processes

This repository contains the code to reproduce the experiments in the paper "Inverse-Free Sparse Variational Gaussian Processes".

## Requirements

The code is written in Python 3.12. The project dependencies are listed in the `pyproject.toml` file. We provide an additional `environment.yaml` file that can be used to create a Conda (or Mamba) environment with all the required dependencies to run the experiments.

## Experiments

### Sections 4.1

We provide two Jupyter notebooks to perform the experiments on the `snelson` and `banana` datasets in Section 4.1 used to obtain Figure 1 in the paper.
The notebooks can be found in the `examples` folder.

### Sections 4.2 and 4.3
We provide Hydra templates to replicate the other experiments in the paper. Experiments can be run with:
```bash
    python relaxedgp_experiments/experiment.py +experiment=<TASK>/<EXPERIMENT_NAME>
```
where `<TASK>` is one of
- `kin40k`: for the shallow GP experiments on the `kin40k` dataset (Section 4.2)
- `elevators`: for the shallow GP experiments on the `elevators` dataset (Section 4.2)
- `kin40k_dgp`: for the deep GP experiments on the `kin40k` dataset (Section 4.3)
- `mnist_dgp`: for the convolutional GP experiments on the `mnist` dataset (Section 4.3)

and where `<EXPERIMENT_NAME>` corresponds to the different models considered in the experiments.
For example, for the `kin40k` task, `<EXPERIMENT_NAME>` can be one of `rsvgp_np_Z_warmup` (R-SVGP), `lsvgp_p_Z` (L-SVGP) and `wsvgp_Z` (W-SVGP).
The possible choices are listed as configuration files in the folder `relaxedgp_experiments/conf/experiment/<TASK>`.
