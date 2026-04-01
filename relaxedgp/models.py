from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes
from gpflow import Parameter, covariances
from gpflow.base import AnyNDArray, InputData, MeanAndVariance, RegressionData
from gpflow.config import default_float, default_jitter
from gpflow.kernels import Kernel, MultioutputKernel, SharedIndependent
from gpflow.likelihoods import Likelihood
from gpflow.mean_functions import MeanFunction
from gpflow.models.model import GPModel
from gpflow.models.training_mixins import ExternalDataTrainingLossMixin
from gpflow.models.util import InducingVariables, inducingpoint_wrapper
from gpflow.utilities import positive, to_default_float, triangular

from .conditionals import likelihood_conditional, relaxed_conditional
from .utils import _K_tilde, cholesky_matmul


class LSVGP(GPModel, ExternalDataTrainingLossMixin):
    """
    Likelihood parameterised SVGP, rougly according to

    ::

      @inproceedings{panos2018fullyscalable,
        title={Fully Scalable Gaussian Processes using Subspace Inducing Inputs},
        author={Panos, Aristeidis and Dellaportas, Aristeidis and Titsias, Michalis K.}
        booktitle={arXiv},
        year={2018}
      }

    """

    @check_shapes(
        "q_mu: [M, P]",
        "q_sqrt: [M] if q_diag",
        "q_sqrt: [M, M] if (not q_diag)",  # TODO: is it ever useful?
    )
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariables,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,  # TODO: extend to multiple latent gps
        q_mu: Optional[tf.Tensor] = None,
        q_sqrt: Optional[tf.Tensor] = None,
        num_data: Optional[tf.Tensor] = None,
        preconditioner: str = "identity",
        q_diag: bool = True,
        with_s2: bool = False,
    ) -> None:
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        self._init_variational_parameters(
            q_mu,
            q_sqrt,
            preconditioner=preconditioner,
            q_diag=q_diag,
            with_s2=with_s2,
        )

    @check_shapes(
        "q_mu: [M]",
        "q_sqrt: [M] if q_diag",
        "q_sqrt: [M, M] if (not q_diag)",  # TODO: is it ever useful?
    )
    def _init_variational_parameters(
        self,
        q_mu: Optional[AnyNDArray],
        q_sqrt: Optional[AnyNDArray],
        preconditioner: str = "identity",
        q_diag: bool = True,
        with_s2: bool = False,
    ) -> None:
        M = self.inducing_variable.num_inducing

        if q_mu is None:
            q_mu = np.zeros((M, self.num_latent_gps))
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if q_diag:
                # NOTE: in this case, q_sqrt actually represents the diagonal of
                # the variational covariance matrix, rather than its Cholesky factor
                # to avoid computing a square
                self.q_sqrt = Parameter(
                    np.ones(M) * 1e-4,
                    dtype=default_float(),
                    transform=positive(),
                )  # [M]
            else:
                self.q_sqrt = Parameter(
                    np.array(np.eye(M)),
                    dtype=default_float(),
                    transform=triangular(),
                )  # [M, M]
        else:
            if q_diag:
                assert q_sqrt.shape[0] == M
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M]
            else:
                assert q_sqrt.shape[0] == M
                assert q_sqrt.shape[1] == M
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [M, M]

        if with_s2:
            self.s2 = Parameter(
                tf.constant(1.0, dtype=default_float()), transform=positive()
            )
            raise NotImplementedError
        else:
            self.s2 = None

        self.preconditioner = preconditioner
        if preconditioner == "identity":
            self.premat = tf.constant(np.eye(M), dtype=default_float())
        elif preconditioner not in ["inv", "T_sqrt", "T"]:
            raise ValueError(f"Unknown preconditioner: {preconditioner}")

    def K_tilde(self, jitter: Optional[float] = None) -> tf.Tensor:
        Kuu = covariances.Kuu(
            self.inducing_variable,
            self.kernel,
            jitter=default_jitter() if jitter is None else jitter,
        )
        return self._K_tilde(Kuu)  # [M, M]

    @check_shapes(
        "Kuu: [M, M]",
        "return: [M, M]",
    )
    def _K_tilde(self, Kuu: tf.Tensor) -> tf.Tensor:
        return _K_tilde(Kuu, self.q_sqrt, None)

    def q_mu_prec(self, L_tilde: Optional[tf.Tensor] = None) -> tf.Tensor:
        preconditioner = self.preconditioner
        q_mu = self.q_mu

        if preconditioner == "identity":
            return q_mu
        elif preconditioner == "inv":
            if L_tilde is None:
                K_tilde = self.K_tilde()
                L_tilde = tf.linalg.cholesky(K_tilde)
            return tf.linalg.cholesky_solve(L_tilde, q_mu)
        elif preconditioner in ["T_sqrt", "T"]:
            raise ValueError(f"LSVGP does not support preconditioner: {preconditioner}")
        else:
            raise ValueError(f"Unknown preconditioner: {preconditioner}")

    @check_shapes(
        "return[0]: [M_M_or_P_M_M...]",
        "return[1]: [M_M_or_P_M_M...]",
        "return[2]: [M_M_or_P_M_M...]",
        "return[3]: [M, P]",
    )
    def _common_calculation(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # NOTE: small jitter is helpful in low precision
        Kuu = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )
        K_tilde = self._K_tilde(Kuu)
        L_tilde = tf.linalg.cholesky(K_tilde)
        q_mu = self.q_mu_prec(L_tilde)

        return Kuu, K_tilde, L_tilde, q_mu

    def prior_kl(self, common: Optional[Tuple] = None) -> tf.Tensor:
        # NOTE: Since q_sqrt is either [M] or [M, M], the size of Kuu and L_tilde
        # always match (i.e. [M, M] or [P, M, M])
        # when q_sqrt will be extended to being different across outputs, this might
        # not hold when Kuu is SharedIndependent and q_sqrt is [P, M, M] or [P, M]

        if common is None:
            common = self._common_calculation()

        Kuu, _, L_tilde, q_mu = common
        q_sqrt = self.q_sqrt  # [M, M] | [M]

        num_func = to_default_float(tf.shape(q_mu)[-1])
        M = tf.shape(q_mu)[-2]

        is_batched = len(Kuu.shape) == 3
        is_diag = len(q_sqrt.shape) == 1

        if is_batched:
            # Kuu: [P, M, M]
            # L_tilde: [P, M, M]

            # Trace term: tr((Kuu + var)^{-1} Kuu)
            K_tilde_inv = tf.linalg.cholesky_solve(
                L_tilde,
                tf.broadcast_to(tf.eye(M, dtype=default_float()), (num_func, M, M)),
            )  # [P, M, M]
            trace = tf.einsum("ijk,ikj", K_tilde_inv, Kuu)  # []

            # Quadratic term: mu^T Kuu mu
            q_mu = tf.transpose(q_mu)  # [P, M]
            quad = tf.reduce_sum(q_mu * tf.linalg.matvec(Kuu, q_mu))  # []

            # Logdet term: log |Kuu + var| - log |var|
            logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_tilde)))
            if is_diag:
                logdet -= num_func * tf.reduce_sum(tf.math.log(q_sqrt))
            else:
                logdet -= num_func * tf.reduce_sum(
                    tf.math.log(tf.square(tf.linalg.diag_part(q_sqrt)))
                )

            return 0.5 * (-trace + quad + logdet)

        # Kuu: [M, M]
        # L_tilde: [M, M]

        trace = tf.linalg.trace(tf.linalg.cholesky_solve(L_tilde, Kuu))  # []

        # Quadratic term: mu^T Kuu mu
        quad = tf.reduce_sum(
            q_mu
            * tf.linalg.matmul(
                Kuu,
                q_mu,
            )
        )  # []

        # Logdet term: log |Kuu + var| - log |var|
        logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L_tilde)))
        if is_diag:
            logdet -= tf.reduce_sum(tf.math.log(q_sqrt))
        else:
            logdet -= tf.reduce_sum(tf.math.log(tf.square(tf.linalg.diag_part(q_sqrt))))

        # NOTE: multiply by num_latent_gps because SharedIndependentKernel
        # might be used
        return 0.5 * (quad + (logdet - trace) * num_func)

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        common = self._common_calculation()
        kl = self.prior_kl(common)
        f_mean, f_var = self.predict_f(
            X, full_cov=False, full_output_cov=False, common=common
        )
        var_exp = self.likelihood.variational_expectations(X, f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)  # kl.dtype
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(
        self,
        Xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
        common: Optional[Tuple] = None,
    ) -> MeanAndVariance:
        mu, var = likelihood_conditional(
            self.kernel,
            self.inducing_variable,
            Xnew=Xnew,
            q_mu=self.q_mu_prec() if common is None else common[-1],
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            common=common,
        )
        return mu + self.mean_function(Xnew), var


class RSVGP(LSVGP):
    """
    Relaxed SVGP
    """

    @check_shapes(
        "q_mu: [M, P]",
        "q_sqrt: [M] if q_diag",
        "q_sqrt: [M, M] if (not q_diag)",  # TODO: is it ever useful?
    )
    def __init__(
        self,
        kernel: Kernel,
        likelihood: Likelihood,
        inducing_variable: InducingVariables,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: int = 1,  # TODO: extend to multiple latent gps
        q_mu: Optional[tf.Tensor] = None,
        q_sqrt: Optional[tf.Tensor] = None,
        num_data: Optional[tf.Tensor] = None,
        q_diag: bool = True,
        num_probes: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            kernel,
            likelihood,
            inducing_variable,
            mean_function=mean_function,
            num_latent_gps=num_latent_gps,
            q_mu=q_mu,
            q_sqrt=q_sqrt,
            num_data=num_data,
            with_s2=False,  # TODO: see note above
            **kwargs,
        )
        self._init_q_T_sqrt(kernel)
        self.num_probes = num_probes

    def _init_q_T_sqrt(self, kernel: Kernel) -> None:
        M = self.inducing_variable.num_inducing
        if isinstance(kernel, MultioutputKernel) and not isinstance(
            kernel, SharedIndependent
        ):
            q_T_sqrt = np.array(
                [np.eye(M) * 1e-3 for _ in range(kernel.num_latent_gps)]
            )  # [P, M, M]
        else:
            q_T_sqrt = np.eye(M) * 1e-3  # [M, M]
        self.q_T_sqrt = Parameter(
            q_T_sqrt,
            dtype=default_float(),
            transform=triangular(),
        )

    def q_mu_prec(self, K_tilde: Optional[tf.Tensor] = None) -> tf.Tensor:
        preconditioner = self.preconditioner
        q_mu = self.q_mu
        q_T_sqrt = self.q_T_sqrt

        if preconditioner == "T_sqrt":
            return tf.linalg.matmul(q_T_sqrt, q_mu, adjoint_a=True)
        elif self.preconditioner == "T":
            if K_tilde is None:
                K_tilde = self.K_tilde()

            # q_mu_prec = T q_mu + T (q_mu - K_tilde T q_mu) = (2 T - T K_tilde T) q_mu
            v = cholesky_matmul(q_T_sqrt, q_mu)

            # might want to cast to float64 before division if unstable
            r = q_mu - tf.matmul(K_tilde, v)

            return v + cholesky_matmul(q_T_sqrt, r)
        else:
            return super().q_mu_prec(K_tilde)

    @check_shapes(
        "return[0]: [M, M]",
        "return[1]: [M, M]",
        "return[2]: [M, P]",
    )
    def _common_calculation(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # NOTE: small jitter is helpful in low precision
        Kuu = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )
        K_tilde = self._K_tilde(Kuu)
        q_mu = self.q_mu_prec(K_tilde)

        return Kuu, K_tilde, q_mu

    def prior_kl(self, common: Optional[Tuple] = None) -> tf.Tensor:
        # NOTE: Since q_sqrt is either [M] or [M, M], the size of Kuu and K_tilde
        # always match (i.e. [M, M] or [P, M, M])
        # when q_sqrt will be extended to being different across outputs, this might
        # not hold when Kuu is SharedIndependent and q_sqrt is [P, M, M] or [P, M]

        if common is None:
            common = self._common_calculation()

        Kuu, K_tilde, q_mu = common
        q_sqrt = self.q_sqrt  # [M, M] | [M]
        q_T_sqrt = self.q_T_sqrt  # [M, M] | [P, M, M]

        num_func = to_default_float(tf.shape(q_mu)[-1])
        M = tf.shape(q_mu)[-2]

        is_batched = len(Kuu.shape) == 3
        is_diag = len(q_sqrt.shape) == 1

        if is_batched:
            # Kuu: [M, M]
            # L_tilde: [M, M]
            # q_T_sqrt: [M, M]

            # TODO: update with self.num_probes
            T = tf.matmul(q_T_sqrt, q_T_sqrt, transpose_b=True)

            # Trace term: tr((T K_tilde T - 2 T) Kuu)
            trace = tf.einsum(
                "ijk,ikj", tf.matmul(T, tf.matmul(K_tilde, T)) - 2.0 * T, Kuu
            )

            # Quadratic term: mu^T Kuu mu
            q_mu = tf.transpose(q_mu)  # [P, M]
            quad = tf.reduce_sum(q_mu * tf.linalg.matvec(Kuu, q_mu))  # []

            # Logdet term: -log |var|
            if is_diag:
                logdet_var = num_func * tf.reduce_sum(tf.math.log(self.q_sqrt))
            else:
                logdet_var = num_func * tf.reduce_sum(
                    tf.math.log(tf.square(tf.linalg.diag_part(self.q_sqrt)))
                )

            # Logdet bound: tr(K_tilde T) - M
            logdet_bound = tf.einsum("ijk,ikj", K_tilde, T) - to_default_float(M)

            logdet_T = tf.reduce_sum(
                tf.math.log(tf.square(tf.linalg.diag_part(self.q_T_sqrt)))
            )
            logdet_bound -= logdet_T

            return 0.5 * (trace + quad + logdet_bound - logdet_var)

        # q_mu: [M, P]
        # Kuu: [M, M]
        # L_tilde: [M, M]
        # q_T_sqrt: [M, M]

        # Quadratic term: mu^T Kuu mu
        quad = tf.reduce_sum(
            q_mu
            * tf.matmul(
                Kuu,
                q_mu,
            )
        )

        # Logdet term: -log |var|
        if is_diag:
            logdet_var = tf.reduce_sum(tf.math.log(self.q_sqrt))
        else:
            logdet_var = tf.reduce_sum(
                tf.math.log(tf.square(tf.linalg.diag_part(self.q_sqrt)))
            )

        # Logdet bound: tr(K_tilde T) - M - log|T|
        logdet_bound = -to_default_float(M) - tf.reduce_sum(
            tf.math.log(tf.square(tf.linalg.diag_part(q_T_sqrt)))
        )

        if self.num_probes is None:
            T = tf.matmul(q_T_sqrt, q_T_sqrt, transpose_b=True)
            TK_tilde = tf.matmul(T, K_tilde)
            TKuu = tf.matmul(T, Kuu)

            logdet_bound += tf.reduce_sum(tf.linalg.diag_part(TK_tilde))

            # Trace term: tr((T K_tilde T - 2 T) Kuu) = tr(T K_tilde T Kuu) -2 tr(T Kuu)
            trace = tf.reduce_sum(tf.transpose(TKuu) * TK_tilde) - 2 * tf.reduce_sum(
                tf.linalg.diag_part(TKuu)
            )
        else:
            r = self.num_probes

            probes = tf.cast(tf.sign(tf.random.uniform((M, r)) - 0.5), default_float())

            a = cholesky_matmul(q_T_sqrt, probes)  # [M, r]
            b = tf.matmul(K_tilde, a)  # [M, r]
            c = tf.matmul(Kuu, probes)  # [M, r]
            d = cholesky_matmul(q_T_sqrt, b)  # [M, r]

            logdet_bound += tf.reduce_sum(probes * b) / r

            trace = tf.reduce_sum(c * d) / r - 2 * tf.reduce_sum(c * a) / r

        # NOTE: multiply by num_latent_gps because SharedIndependentKernel
        # might be used
        return 0.5 * (quad + (trace + logdet_bound - logdet_var) * num_func)

    def predict_f(
        self,
        Xnew: tf.Tensor,
        full_cov: bool = False,
        full_output_cov: bool = False,
        common: Optional[Tuple] = None,
    ) -> MeanAndVariance:
        mu, var = relaxed_conditional(
            self.kernel,
            self.inducing_variable,
            Xnew=Xnew,
            q_mu=self.q_mu_prec() if common is None else common[-1],
            q_sqrt=self.q_sqrt,
            q_T_sqrt=self.q_T_sqrt,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            common=common,
        )
        return mu + self.mean_function(Xnew), var

    # TODO: extend to multi-output
    def quality_T(self, jitter: Optional[float] = None) -> tf.Tensor:
        """
        Here, we compute a metric for the quality of T.
        Specifically, the bound on log |Kuu + q_sqrt|.
        """
        M = self.inducing_variable.num_inducing
        K_tilde = self.K_tilde(jitter=default_float() if jitter is None else jitter)
        L = tf.linalg.cholesky(K_tilde)
        T = tf.matmul(self.q_T_sqrt, self.q_T_sqrt, transpose_b=True)

        logdet_T = tf.reduce_sum(
            tf.math.log(tf.square(tf.linalg.diag_part(self.q_T_sqrt)))
        )

        logdet_K_tilde = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        trace = tf.reduce_sum(K_tilde * T) - to_default_float(M)
        return trace - logdet_T - logdet_K_tilde

    def set_optimal_T(self, jitter: Optional[float] = None) -> None:
        K_tilde = self.K_tilde(jitter=default_float() if jitter is None else jitter)
        K_tilde_inv = tf.linalg.inv(K_tilde)
        self.q_T_sqrt.assign(tf.linalg.cholesky(K_tilde_inv))
