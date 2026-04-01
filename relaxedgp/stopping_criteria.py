from abc import abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import tensorflow as tf
from check_shapes import check_shapes
from gpflow import covariances
from gpflow.base import RegressionData, TensorType

from .models import RSVGP


class StoppingCriterion:
    def __init__(
        self,
        tol: float,
        num_data: Optional[tf.Tensor] = None,
        stop_on_plateau_patience: Optional[int] = None,
        stop_on_plateau_min_delta: float = 0.0,
    ) -> None:
        self.tol = tol
        self.num_data = num_data

        if stop_on_plateau_patience is not None:
            self.stop_on_plateau = StopOnPlateau(
                stop_on_plateau_patience,
                stop_on_plateau_min_delta,
            )
        else:
            self.stop_on_plateau = None

    @abstractmethod
    def __call__(
        self,
        A: TensorType,
        B: TensorType,
        *,
        model: Optional[RSVGP] = None,
        data: Optional[RegressionData] = None,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[tf.Tensor, bool], Tuple[tf.Tensor, bool, Dict]]:
        raise NotImplementedError

    def _precompute(
        self, *, A: Optional[TensorType] = None, model: Optional[RSVGP] = None
    ) -> None:
        pass


class FrobeniusNormCriterion(StoppingCriterion):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.M_sqrt = None

    @check_shapes(
        "A: [M, M]",
        "B: [M, M]",
    )
    def __call__(
        self,
        A: TensorType,
        B: TensorType,
        *,
        model: Optional[RSVGP] = None,
        data: Optional[RegressionData] = None,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[tf.Tensor, bool], Tuple[tf.Tensor, bool, Dict]]:
        M = A.shape[0]
        AB = tf.matmul(A, B)
        BtAB = tf.matmul(B, AB, transpose_a=True)

        if self.M_sqrt is None:
            self._precompute(A=A)

        criterion = (
            tf.norm(tf.cast(BtAB, tf.float64) - tf.eye(M, dtype=tf.float64))
            / self.M_sqrt
        )

        is_converged = criterion < self.tol

        if self.stop_on_plateau is not None:
            is_converged = self.stop_on_plateau(criterion, is_converged)

        if return_aux:
            return criterion, is_converged, {"BtAB": BtAB}
        return criterion, is_converged

    def _precompute(self, *, A: TensorType, model: Optional[RSVGP] = None) -> None:
        self.M_sqrt = tf.sqrt(tf.cast(tf.shape(A)[0], dtype=tf.float64))


class PredictiveVarianceCriterion(StoppingCriterion):
    def __init__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.min_q_sqrt = None

    @check_shapes(
        "A: [M, M]",
        "B: [M, M]",
    )
    def __call__(
        self,
        A: TensorType,
        B: TensorType,
        *,
        model: RSVGP,
        data: RegressionData,
        return_aux: bool = False,
        **kwargs: Any,
    ) -> Union[Tuple[tf.Tensor, bool], Tuple[tf.Tensor, bool, Dict]]:
        X, _ = data

        Kun = covariances.Kuf(model.inducing_variable, model.kernel, X)

        M, N_batch = Kun.shape

        if self.min_q_sqrt is None:
            self._precompute(model=model)

        # criterion:  ||(I - K_tilde T) Kun||^2 / \sigma^2
        AB = tf.matmul(A, B)
        criterion = tf.reduce_sum(
            tf.square(
                tf.cast(Kun, tf.float64)
                - tf.cast(
                    tf.matmul(
                        AB,
                        tf.matmul(B, Kun, transpose_a=True),
                    ),
                    tf.float64,
                )
            )
        ) / tf.cast(
            tf.reduce_min(model.q_sqrt), tf.float64
        )  # TODO: this only works when q_diag is True

        scale = 1.0 / N_batch  # impact on elbo per datapoint

        criterion *= scale
        tol = 2 * tf.cast(model.likelihood.variance, tf.float64) * self.tol
        is_converged = criterion < tol

        if self.stop_on_plateau is not None and not is_converged:
            is_converged = self.stop_on_plateau(criterion, is_converged)

        if return_aux:
            return criterion, is_converged, {"AB": AB}
        return criterion, is_converged

    def _precompute(self, *, model: RSVGP, A: Optional[TensorType] = None) -> None:
        self.min_q_sqrt = tf.cast(tf.reduce_min(model.q_sqrt), tf.float64)


class StopOnPlateau:
    def __init__(
        self, patience: int, min_delta: float = 0.0, on_device: Optional[str] = None
    ) -> None:
        self.patience = tf.constant(int(patience), tf.int32)
        self.min_delta = tf.cast(min_delta, tf.float64)
        self._on_device = on_device
        self._build()

    def _build(self) -> None:
        with tf.init_scope():
            if self._on_device is not None:
                device = self._on_device
            else:
                gpus = tf.config.list_logical_devices("GPU")
                device = gpus[0].name if gpus else "/CPU:0"

            with tf.device(device):
                self._best = tf.Variable(
                    tf.constant(float("inf"), tf.float64), trainable=False, name="best"
                )
                # for some reason tf.int64 vs tf.int32 is needed to force XLA to put
                # the device on GPU if available...
                self._stalled = tf.Variable(
                    tf.constant(0, dtype=tf.int64), trainable=False, name="stalled"
                )

    def _reset(self) -> None:
        self._best.assign(tf.constant(float("inf"), tf.float64))
        self._stalled.assign(0)

    def __call__(self, criterion: float, is_converged: bool):
        # cast back to int32 if needed
        _stalled = tf.cast(self._stalled, dtype=self.patience.dtype)

        improved = criterion < (self._best - self.min_delta)

        self._best.assign(tf.where(improved, criterion, self._best))
        self._stalled.assign(
            tf.cast(tf.where(improved, 0, _stalled + 1), dtype=self._stalled.dtype),
        )

        plateau = tf.logical_and(
            tf.greater(self.patience, 0),
            tf.greater_equal(_stalled, self.patience),
        )

        is_converged = tf.logical_or(is_converged, plateau)

        if is_converged:
            self._reset()

        return is_converged
