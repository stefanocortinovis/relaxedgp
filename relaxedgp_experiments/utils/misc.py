import os
import random
from collections.abc import Iterator
from typing import Any, Dict, Tuple

import gpflow
import numpy as np
import tensorflow as tf


def _flatten_dict(d: Dict[str, Any], parent_key: str = "") -> Iterator[Tuple[str, Any]]:
    for k, v in d.items():
        new_key = ".".join(filter(None, [parent_key, k]))
        if isinstance(v, dict):
            yield from _flatten_dict(v, new_key)
        else:
            yield new_key, v


def flatten_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    return dict(_flatten_dict(d))


def set_default_float(float_type: str = "float64"):
    if float_type == "float64":
        gpflow.config.set_default_float(tf.float64)
    elif float_type == "float32":
        gpflow.config.set_default_float(tf.float32)
    elif float_type == "float16":
        gpflow.config.set_default_float(tf.float16)
    else:
        raise ValueError(f"Unknown float type: {float_type}")


def set_seeds(seed: int = 0):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
