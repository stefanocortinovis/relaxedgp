import logging
from typing import Callable, Optional

from gpflux.models import DeepGP


def get_callback(
    model: DeepGP,
    evaluate_func,
    evaluate_every: int = 1,
    logger: Optional[logging.Logger] = None,
    log_wandb: bool = False,
    n_iter: Optional[int] = None,
) -> Callable:
    step = 0

    def _callback():
        nonlocal step

        metrics = dict()
        if (step == 0) or ((step + 1) % evaluate_every == 0) or (step == n_iter - 1):
            metrics.update(evaluate_func())

        step += 1

        return metrics

    return _callback
