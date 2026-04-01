from typing import Union, cast

import gpflow
import numpy as np
import tensorflow as tf
from gpflow.base import Parameter
from gpflow.kernels import ArcCosine, SquaredExponential
from gpflow.likelihoods import Bernoulli, Gaussian, MultiClass, RobustMax
from gpflux.helpers import (
    construct_basic_inducing_variables,
    construct_basic_kernel,
    construct_mean_function,
)
from gpflux.layers.likelihood_layer import LikelihoodLayer
from gpflux.models import DeepGP
from scipy.cluster.vq import kmeans2

from ..typing import MinibatchModel
from .layers import GPLayerWrapper


def build_constant_input_dim_deep_gp(
    base_model_class: MinibatchModel,
    X: np.ndarray,
    num_layers: int,
    *,
    num_inducing: int,
    inner_layer_qsqrt_factor: float = 1.0,
    likelihood_noise_variance: float = 1.0,
    hidden_kernel_variance: float = 1e-6,
    task: str = "regression",
    kernel_name: str = "squared_exponential",
    **kwargs,
) -> DeepGP:
    if X.dtype != gpflow.default_float():
        raise ValueError(
            "X needs to have dtype according to gpflow.default_float() = "
            f"{gpflow.default_float()} however got X with {X.dtype} dtype."
        )

    num_data, input_dim = X.shape
    X_running = X

    gp_layers = []
    centroids, _ = kmeans2(X, k=num_inducing, minit="points")

    if "classification_multi" in task:
        _, _, num_classes = task.split("_")
        num_classes = int(num_classes)
        D_out_last = num_classes
    else:
        D_out_last = 1

    for i_layer in range(num_layers):
        is_last_layer = i_layer == num_layers - 1
        D_in = input_dim
        D_out = D_out_last if is_last_layer else input_dim

        inducing_var = construct_basic_inducing_variables(
            num_inducing=num_inducing,
            input_dim=D_in,
            share_variables=True,
            z_init=Parameter(centroids, name="inducing_loc"),
        )

        kernel = construct_basic_kernel(
            kernels=_construct_kernel(
                kernel_name, D_in, is_last_layer, hidden_kernel_variance
            ),
            output_dim=D_out,
            share_hyperparams=True,
        )

        if is_last_layer:
            mean_function = gpflow.mean_functions.Zero()
            q_sqrt_scaling = 1.0
        else:
            mean_function = construct_mean_function(X_running, D_in, D_out)
            X_running = mean_function(X_running)
            if tf.is_tensor(X_running):
                X_running = cast(tf.Tensor, X_running).numpy()

        # seems to be working well, but recheck
        q_sqrt_scaling = inner_layer_qsqrt_factor

        layer = GPLayerWrapper(
            base_model_class(
                kernel=kernel,
                likelihood=None,
                inducing_variable=inducing_var,
                mean_function=mean_function,
                num_data=num_data,
                num_latent_gps=D_out,
                **kwargs,
            ),
        )
        layer.model.q_sqrt.assign(layer.model.q_sqrt * q_sqrt_scaling)
        gp_layers.append(layer)

    if task == "regression":
        likelihood = Gaussian(likelihood_noise_variance)
    elif task == "classification":
        likelihood = Bernoulli()
    elif "classification_multi" in task:
        invlink = RobustMax(D_out_last)
        likelihood = MultiClass(D_out_last, invlink=invlink)
    else:
        raise ValueError(f"Unknown task {task}")

    return DeepGP(gp_layers, LikelihoodLayer(likelihood))


def _construct_kernel(
    kernel_name: str,
    input_dim: int,
    is_last_layer: bool,
    hidden_kernel_variance: float = 1e-6,
) -> Union[ArcCosine, SquaredExponential]:
    variance = hidden_kernel_variance if not is_last_layer else 1.0

    if kernel_name == "squared_exponential":
        lengthscales = [2.0] * input_dim
        # lengthscales = [np.sqrt(input_dim)] * input_dim  # used by hugh
        return SquaredExponential(lengthscales=lengthscales, variance=variance)
    elif kernel_name == "arc_cosine_0":
        return ArcCosine(order=0, variance=variance)
    elif kernel_name == "arc_cosine_1":
        return ArcCosine(order=1, variance=variance)
    raise ValueError("Unknown kernel: {kernel_name}.")
