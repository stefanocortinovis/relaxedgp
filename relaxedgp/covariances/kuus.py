import tensorflow as tf
from check_shapes import check_shapes
from gpflow.covariances.dispatch import Kuu
from gpflow.inducing_variables import InducingPatches

from ..kernels import WhitenedConvolutional


@Kuu.register(InducingPatches, WhitenedConvolutional)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "return: [M, M]",
)
def Kuu_whiteconv_patch(
    inducing_variable: InducingPatches,
    kernel: WhitenedConvolutional,
    jitter: float = 0.0,
) -> tf.Tensor:
    return Kuu(
        inducing_variable, kernel.convolutional_kernel, jitter=jitter
    ) + kernel.white_kernel.K_diag(inducing_variable.Z)
