from typing import Optional, Tuple

import tensorflow as tf
from check_shapes import check_shapes
from gpflow import covariances
from gpflow.base import MeanAndVariance
from gpflow.conditionals.util import expand_independent_outputs
from gpflow.inducing_variables import (
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import SeparateIndependent, SharedIndependent

from ..utils import _K_tilde
from .base import (
    _base_likelihood_conditional,
    _base_likelihood_conditional_separate_independent,
    _base_relaxed_conditional,
    _base_relaxed_conditional_separate_independent,
)
from .dispatch import likelihood_conditional, relaxed_conditional


@likelihood_conditional.register(
    (SharedIndependent, SeparateIndependent),
    (SharedIndependentInducingVariables, SeparateIndependentInducingVariables),
)
@check_shapes(
    "inducing_variable: [M, D, broadcast P]",  # TODO: L instead of P?
    "Xnew: [batch..., N, D]",
    "q_mu: [M, P]",  # TODO: L instead of P?
    "q_sqrt: [M_or_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def _likelihood_conditional_independent_mo(
    kernel: SharedIndependent,
    inducing_variable: SharedIndependentInducingVariables,
    *,
    Xnew: tf.Tensor,
    q_mu: tf.Tensor,
    q_sqrt: tf.Tensor,
    full_cov: bool = False,
    full_output_cov: bool = False,
    common: Optional[Tuple] = None,
) -> MeanAndVariance:
    if common is not None:
        _, K_tilde, L_tilde, _ = common
    else:
        Kuu = covariances.Kuu(inducing_variable, kernel, jitter=0.0)
        K_tilde = _K_tilde(Kuu, q_sqrt)
        L_tilde = tf.linalg.cholesky(K_tilde)  # [M, M] | [P, M, M]

    Kun = covariances.Kuf(
        inducing_variable, kernel, Xnew
    )  # [M, ..., N] | [P, M, ..., N]

    if isinstance(inducing_variable, SharedIndependentInducingVariables) and isinstance(
        kernel, SharedIndependent
    ):
        Knn = kernel.kernel(Xnew, full_cov=full_cov)  # [..., N, N] | [..., N]

        fmean, fvar = _base_likelihood_conditional(
            Kun, L_tilde, Knn, q_mu, full_cov=full_cov
        )
    else:
        if isinstance(kernel, SeparateIndependent):
            kernel_list = kernel.kernels
        else:
            kernel_list = [kernel.kernel] * len(
                inducing_variable.inducing_variable_list
            )

            Knn = tf.stack(
                [k.K(Xnew) if full_cov else k.K_diag(Xnew) for k in kernel_list], axis=0
            )  # [P, ..., N, N] | [P, ..., N]

            fmean, fvar = _base_likelihood_conditional_separate_independent(
                Kun, L_tilde, Knn, q_mu, full_cov=full_cov
            )

    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)


@relaxed_conditional.register(
    (SharedIndependent, SeparateIndependent),
    (SharedIndependentInducingVariables, SeparateIndependentInducingVariables),
)
@check_shapes(
    "inducing_variable: [M, D, broadcast P]",  # TODO: L instead of P?
    "Xnew: [batch..., N, D]",
    "q_mu: [M, P]",  # TODO: L instead of P?
    "q_sqrt: [M_or_M_M...]",
    "q_T_sqrt: [M_M_or_P_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def _relaxed_conditional_independent_mo(
    kernel: SharedIndependent,
    inducing_variable: SharedIndependentInducingVariables,
    *,
    Xnew: tf.Tensor,
    q_mu: tf.Tensor,
    q_sqrt: tf.Tensor,
    q_T_sqrt: Optional[tf.Tensor],
    full_cov: bool = False,
    full_output_cov: bool = False,
    common: Optional[Tuple] = None,
) -> MeanAndVariance:
    if common is not None:
        _, K_tilde, _ = common
    else:
        Kuu = covariances.Kuu(inducing_variable, kernel, jitter=0.0)
        K_tilde = _K_tilde(Kuu, q_sqrt)  # [M, M] | [P, M, M]

    Kun = covariances.Kuf(
        inducing_variable, kernel, Xnew
    )  # [M, ..., N] | [P, M, ..., N]

    if isinstance(inducing_variable, SharedIndependentInducingVariables) and isinstance(
        kernel, SharedIndependent
    ):
        Knn = kernel.kernel(Xnew, full_cov=full_cov)  # [..., N, N] | [..., N]

        fmean, fvar = _base_relaxed_conditional(
            Kun, K_tilde, Knn, q_mu, q_T_sqrt, full_cov=full_cov
        )
    else:
        if isinstance(kernel, SeparateIndependent):
            kernel_list = kernel.kernels
        else:
            kernel_list = [kernel.kernel] * len(
                inducing_variable.inducing_variable_list
            )

            Knn = tf.stack(
                [k.K(Xnew) if full_cov else k.K_diag(Xnew) for k in kernel_list], axis=0
            )  # [P, ..., N, N] | [P, ..., N]

            fmean, fvar = _base_relaxed_conditional_separate_independent(
                Kun, K_tilde, Knn, q_mu, q_T_sqrt, full_cov=full_cov
            )

    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)
