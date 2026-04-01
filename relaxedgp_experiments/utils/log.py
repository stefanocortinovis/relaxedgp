import logging
from typing import Callable, Dict, Optional

import wandb
from gpflow.likelihoods import Gaussian
from gpflow.utilities.traversal import _get_leaf_components
from scipy.optimize import OptimizeResult

from relaxedgp.models import LSVGP, RSVGP
from relaxedgp.typing import FullBatchModel, Model


def get_callback(
    model: Model,
    evaluate_func,
    evaluate_every: int = 1,
    logger: Optional[logging.Logger] = None,
    log_wandb: bool = False,
    log_quality_T: bool = False,
    n_iter: Optional[int] = None,
    debug_force_bfgs: bool = False,
    debug_log_all_T: bool = False,
) -> Callable:
    step = 0

    if isinstance(model, FullBatchModel) or debug_force_bfgs:

        def _callback(intermediate_result: OptimizeResult):
            nonlocal step

            metrics = {"train/loss": intermediate_result.fun}
            if (
                (step == 0)
                or ((step + 1) % evaluate_every == 0)
                or hasattr(intermediate_result, "success")
            ):
                metrics.update(get_params(model))
                metrics.update(evaluate_func())
                log_metrics(metrics, step=step + 1, logger=logger, log_wandb=log_wandb)
            elif log_wandb:
                log_metrics(metrics, step=step + 1, logger=logger, log_wandb=log_wandb)

            step += 1
    else:

        def _callback():
            nonlocal step

            metrics = dict()
            if (
                (step == 0)
                or ((step + 1) % evaluate_every == 0)
                or (step == n_iter - 1)
            ):
                metrics.update(evaluate_func())

                if isinstance(model, RSVGP) and log_quality_T and not debug_log_all_T:
                    metrics["quality_T"] = model.quality_T().numpy().item()

            if isinstance(model, RSVGP) and debug_log_all_T:
                metrics["quality_T"] = model.quality_T().numpy().item()

            step += 1

            return metrics

    return _callback


def get_params(
    model,
):
    params = {}
    if isinstance(model.likelihood, Gaussian):
        params["variance"] = model.likelihood.variance.numpy().item()

    for name, value in _get_leaf_components(model.kernel).items():
        value = value.numpy()
        if value.ndim > 0:
            for i, v in enumerate(value):
                params[f"params/{name}_{i}"] = v
        else:
            params[f"params/{name}"] = value.item()
    if isinstance(model, LSVGP):
        params["q_mu"] = model.q_mu.numpy().mean().item()
        params["q_sqrt"] = model.q_sqrt.numpy().mean().item()

        if model.s2 is not None:
            params["s2"] = model.s2.numpy().mean().item()

    return params


def log_metrics(
    metrics: Dict[str, float],
    step: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    log_wandb: bool = False,
) -> None:
    if log_wandb:
        wandb.log(metrics, step=step)
    else:
        if logger is None:
            raise ValueError("logger must be provided if not logging to wandb.")
        metrics = [f"{name}: {value:.4f}" for name, value in metrics.items()]
        if step is not None:
            metrics = [f"Step {step: <5}"] + metrics
        logger.info("\t".join(metrics))
