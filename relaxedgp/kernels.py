import tensorflow as tf
from gpflow import Parameter
from gpflow.kernels import Convolutional, IsotropicStationary, Kernel, White
from gpflow.utilities import to_default_float
from tensorflow_probability import bijectors


class GammaExponential(IsotropicStationary):
    """
    The gamma exponential, or γ-exponential kernel. The kernel equation is

        k(r) =  σ² exp{-½ rˠ}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscale
        parameter ℓ.
    γ   controls the roughness of the process.

    The process is equivalent to the squared exponential for γ=2, and nowhere
    differentiable for other values. This is slightly strange, especially since samples
    do look smoother as γ approaches 2. See Rasmussen & Williams p86.
    """

    def __init__(
        self, variance=1.0, lengthscales=1.0, gamma=1.1, active_dims=None, name=None
    ):
        super().__init__(variance, lengthscales, active_dims=active_dims, name=name)
        self.gamma = Parameter(
            gamma,
            transform=bijectors.Sigmoid(to_default_float(1.0), to_default_float(2.0)),
        )

    def K_r(self, r: tf.Tensor) -> tf.Tensor:
        return self.variance * tf.exp(-0.5 * r**self.gamma)


class WhitenedConvolutional(Convolutional):
    def __init__(
        self,
        convolutional_kernel: Convolutional,
        white_variance: float = 1.0,
    ) -> None:
        super(Kernel, self).__init__()
        self.convolutional_kernel = convolutional_kernel
        self.white_kernel = White(white_variance)

    def K(self, X, X2) -> tf.Tensor:
        if X2 is not None:
            self.convolutional_kernel.K(X, X2)
        return self.convolutional_kernel.K(X, X2) + self.white_kernel.K(X, X2)

    def K_diag(self, X) -> tf.Tensor:
        return self.convolutional_kernel.K_diag(X) + self.white_kernel.K_diag(X)

    def slice(self, X, X2):
        return self.convolutional_kernel.slice(X, X2)
