import numpy as np
import tensorflow as tf
from gpflow import Parameter
from gpflow.config import default_float, default_jitter
from gpflow.kernels import SquaredExponential
from gpflow.utilities import triangular

from relaxedgp.natgrad import NaturalGradientInverseCholesky
from relaxedgp.schedules import ConstantSchedule


def test_natgrad_invert() -> None:
    N, D = 43, 2
    X = np.random.randn(N, D)
    kernel = SquaredExponential(variance=0.34, lengthscales=[1.27, 1.35])

    K = kernel.K(X) + default_jitter() * tf.eye(N, dtype=default_float())

    q_T_sqrt = Parameter(
        np.eye(N) * 1e-3,
        dtype=default_float(),
        transform=triangular(),
    )

    opt = NaturalGradientInverseCholesky(
        gamma=ConstantSchedule(max_steps=50, value=1.0)
    )
    opt.minimize(K, q_T_sqrt)

    np.testing.assert_allclose(
        tf.linalg.inv(K),
        tf.matmul(q_T_sqrt, q_T_sqrt, transpose_b=True),
        rtol=1e-5,
        atol=1e-5,
    )
