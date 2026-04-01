from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from check_shapes import check_shapes
from gpflow.base import Parameter, RegressionData, TensorType
from gpflow.config import default_float
from gpflow.optimizers.natgrad import Scalar

from .models import RSVGP
from .schedules import ConstantSchedule, NaturalGradientSchedule


class NaturalGradientInverseCholesky:
    """
    Implements a natural gradient descent optimizer for finding the Cholesky
    factor of the inverse of a matrix A. The iterative update is obtained by
    applying natural gradient descent to the loss function

        KL[N(0, B B^T) || N(0, A^-1)]

    with respect to a lower triangular matrix B.
    """

    def __init__(
        self,
        gamma: Optional[NaturalGradientSchedule] = None,
        name: Optional[str] = None,
    ) -> None:
        """
        :param gamma: natgrad step length
        :param name: optional name for the optimizer
        """
        self.name = self.__class__.__name__ if name is None else name

        if gamma is None:
            gamma = ConstantSchedule(max_steps=1, value=1.0)
        self.gamma = gamma

    @check_shapes(
        "A: [M_or_M_M...]",
        "B: [M_or_M_M...]",
    )
    def minimize(
        self,
        A: TensorType,
        B: Parameter,
        model: Optional[RSVGP] = None,
        data: Optional[RegressionData] = None,
        init: Optional[TensorType] = None,
    ) -> None:
        """
        Finds the Cholesky factor of the inverse of A by minimising

                KL[N(0, B B^T) || N(0, A^-1)]

        with respect to B using natural gradient descent.

        :param A: positive definite matrix to invert
        :param B: lower triangular matrix to optimise
        """

        # TODO: perform operations more efficiently by exploiting fact
        # that B is lower triangular
        if init is None:
            # same as tf.convert_to_tensor?
            B_new = B.bijector(B.unconstrained_variable)
        else:
            B_new = init

        if self.gamma.stopping_criterion is not None:
            self.gamma.stopping_criterion._precompute(A=A, model=model)
            if isinstance(self.gamma, ConstantSchedule) and self.gamma.line_search:
                has_converged = tf.constant(False)
                best_criterion = tf.constant(np.inf, dtype=default_float())
                gamma = self.gamma(0)
                B_prop = B_new
                for i in tf.range(self.gamma.max_steps):
                    criterion, has_converged, aux = self.gamma.stopping_criterion(
                        A,
                        B_prop,
                        model=model,
                        data=data,
                        return_aux=True,
                    )

                    # if convergence reached or criterion gets worse
                    if has_converged:
                        B_new = B_prop
                        break
                    if criterion > best_criterion:
                        B_new = B_prop
                    elif criterion < best_criterion:
                        best_criterion = criterion
                        B_new = B_prop

                    B_prop, _ = self._minimize(A, B_prop, gamma, **aux)
                    gamma = tf.clip_by_value(gamma * 2.0, 0.0, 1.0)
            else:
                for i in tf.range(self.gamma.max_steps):
                    criterion, has_converged, aux = self.gamma.stopping_criterion(
                        A,
                        B_new,
                        model=model,
                        data=data,
                        return_aux=True,
                    )
                    if has_converged:
                        break
                    B_new, _ = self._minimize(A, B_new, self.gamma(i), **aux)
        else:
            for i in tf.range(self.gamma.max_steps):
                B_new, _ = self._minimize(A, B_new, self.gamma(i))

            criterion = None

        B.assign(B_new)

        return i + 1, criterion

    @check_shapes(
        "A: [M_or_M_M...]",
        "B: [M_or_M_M...]",
    )
    def _minimize(
        self,
        A: TensorType,
        B: TensorType,
        gamma: Scalar,
        *,
        nat_dL_dB: Optional[TensorType] = None,
        BtAB: Optional[TensorType] = None,
        AB: Optional[TensorType] = None,
    ) -> Tuple[TensorType, TensorType]:
        # TODO: perform operations more efficiently by exploiting fact
        # that B is lower triangular
        if nat_dL_dB is None:
            nat_dL_dB = self._nat_dL_dB(A, B, BtAB=BtAB, AB=AB)
        return B - gamma * nat_dL_dB, nat_dL_dB

    @check_shapes(
        "A: [M_or_M_M...]",
        "B: [M_or_M_M...]",
    )
    def _nat_dL_dB(
        self,
        A: TensorType,
        B: TensorType,
        *,
        BtAB: Optional[TensorType] = None,
        AB: Optional[TensorType] = None,
    ) -> TensorType:
        if BtAB is None:
            if AB is None:
                AB = tf.matmul(A, B)
            BtAB = tf.matmul(B, AB, transpose_a=True)
        nat_dL_dB = tf.linalg.band_part(BtAB, -1, 0)
        nat_dL_dB = tf.linalg.set_diag(
            nat_dL_dB,
            0.5 * tf.linalg.diag_part(nat_dL_dB) - 0.5,
        )
        return tf.matmul(B, nat_dL_dB)
