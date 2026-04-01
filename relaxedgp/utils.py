from typing import Optional

import tensorflow as tf


def _K_tilde(
    Kuu: tf.Tensor, q_sqrt: tf.Tensor, s2: Optional[tf.Tensor] = None
) -> tf.Tensor:
    if q_sqrt.shape.ndims == 1:  # q_diag is True
        K_tilde = tf.linalg.set_diag(
            Kuu,
            tf.linalg.diag_part(Kuu) + q_sqrt,
        )
    elif q_sqrt.shape.ndims == 2:
        K_tilde = Kuu + tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
    else:
        raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

    if s2 is not None:
        return tf.linalg.set_diag(
            K_tilde,
            tf.linalg.diag_part(K_tilde) + s2,
        )
    return K_tilde


def cholesky_matmul(
    chol,  # [M, M]
    y,  # [M, P]
):
    return tf.linalg.matmul(
        chol,
        tf.linalg.matmul(chol, y, transpose_a=True),
    )
