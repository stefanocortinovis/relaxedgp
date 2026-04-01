import tensorflow as tf
from check_shapes import check_shapes
from gpflow.base import TensorType
from gpflow.covariances import Kuf
from gpflow.inducing_variables import InducingPatches

from ..kernels import WhitenedConvolutional


@Kuf.register(InducingPatches, WhitenedConvolutional, object)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D2]",
    "return: [M, batch..., N]",
)
def Kuf_whiteconv_patch(
    inducing_variable: InducingPatches, kernel: WhitenedConvolutional, Xnew: TensorType
) -> tf.Tensor:
    return Kuf(inducing_variable, kernel.convolutional_kernel, Xnew)
