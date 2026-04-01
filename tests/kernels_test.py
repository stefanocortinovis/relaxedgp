import numpy as np
from gpflow.kernels import SquaredExponential

from relaxedgp.kernels import GammaExponential


def test_gamma_exponential():
    se = SquaredExponential(variance=0.34, lengthscales=1.27)
    ge = GammaExponential(variance=0.34, lengthscales=1.27, gamma=1.9999)

    X = np.random.randn(43, 2)

    seK = se.K(X).numpy()
    geK = ge.K(X).numpy()

    np.testing.assert_allclose(geK, seK, rtol=1e-5, atol=1e-5)
