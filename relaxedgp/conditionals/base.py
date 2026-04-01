import tensorflow as tf
from check_shapes import check_shapes
from gpflow.base import MeanAndVariance


@check_shapes(
    "Kun: [M, batch..., N]",
    "L_tilde: [M, M]",
    "Knn: [batch..., N, N] if full_cov",
    "Knn: [batch..., N] if not full_cov",
    "q_mu: [M, P]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., P, N, N] if full_cov",
    "return[1]: [batch..., N, P] if not full_cov",
)
def _base_likelihood_conditional(
    Kun: tf.Tensor,
    L_tilde: tf.Tensor,
    Knn: tf.Tensor,
    q_mu: tf.Tensor,
    *,
    full_cov: bool = False,
) -> MeanAndVariance:
    num_func = tf.shape(q_mu)[-1]
    M = tf.shape(q_mu)[-2]
    N = tf.shape(Kun)[-1]

    K = tf.rank(Kun)
    perm = tf.concat(
        [
            tf.range(1, K - 1),  # leading dims (...)
            (0, K - 1),  # [M, N]
        ],
        0,
    )  # [N]
    Kun = tf.transpose(Kun, perm)  # [..., M, N]
    leading_dims = tf.shape(Kun)[:-2]

    L_tilde = tf.broadcast_to(
        L_tilde,
        tf.concat([leading_dims, (M, M)], 0),
    )  # [..., M, M]

    q_mu = tf.broadcast_to(
        q_mu, tf.concat([leading_dims, (M, num_func)], 0)
    )  # [..., M, P]

    fmean = tf.matmul(Kun, q_mu, transpose_a=True)  # [..., N, P]

    A = tf.linalg.triangular_solve(L_tilde, Kun)  # [..., M, N]

    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)  # [..., N, N]
        fvar = tf.broadcast_to(
            tf.expand_dims(fvar, -3),
            tf.concat([leading_dims, (num_func, N, N)], 0),
        )  # [..., P, N, N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
        fvar = tf.broadcast_to(
            tf.expand_dims(fvar, -1),
            tf.concat([leading_dims, (N, num_func)], 0),
        )  # [..., N, P]

    return fmean, fvar


@check_shapes(
    "Kun: [M, batch..., N]",
    "K_tilde: [M, M]",
    "Knn: [batch..., N, N] if full_cov",
    "Knn: [batch..., N] if not full_cov",
    "q_mu: [M, P]",
    "q_T_sqrt: [M, M]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., P, N, N] if full_cov",
    "return[1]: [batch..., N, P] if not full_cov",
)
def _base_relaxed_conditional(
    Kun: tf.Tensor,
    K_tilde: tf.Tensor,
    Knn: tf.Tensor,
    q_mu: tf.Tensor,
    q_T_sqrt: tf.Tensor,
    *,
    full_cov: bool = False,
) -> MeanAndVariance:
    num_func = tf.shape(q_mu)[-1]
    M = tf.shape(q_mu)[-2]
    N = tf.shape(Kun)[-1]

    K = tf.rank(Kun)
    perm = tf.concat(
        [
            tf.range(1, K - 1),  # leading dims (...)
            (0, K - 1),  # [M, N]
        ],
        0,
    )  # [N]
    Kun = tf.transpose(Kun, perm)  # [..., M, N]
    leading_dims = tf.shape(Kun)[:-2]

    K_tilde = tf.broadcast_to(
        K_tilde,
        tf.concat([leading_dims, (M, M)], 0),
    )  # [..., M, M]

    q_T_sqrt = tf.broadcast_to(
        q_T_sqrt,
        tf.concat([leading_dims, (M, M)], 0),
    )  # [..., M, M]

    q_mu = tf.broadcast_to(
        q_mu, tf.concat([leading_dims, (M, num_func)], 0)
    )  # [..., M, P]

    fmean = tf.matmul(Kun, q_mu, transpose_a=True)  # [..., N, P]

    # This is better if minibatch is small.
    T_Kun = tf.matmul(
        q_T_sqrt, tf.matmul(q_T_sqrt, Kun, transpose_a=True)
    )  # [..., M, N]

    # fvar = Knn + Kun^T * (T K_tilde T - 2T) Kun
    #   = Knn + Kun^T T (K_tilde T - 2I) Kun
    if full_cov:
        fvar = (
            Knn
            + tf.matmul(T_Kun, tf.matmul(K_tilde, T_Kun), transpose_a=True)
            - 2 * tf.matmul(Kun, T_Kun, transpose_a=True)
        )
        fvar = tf.broadcast_to(
            tf.expand_dims(fvar, -3),
            tf.concat([leading_dims, (num_func, N, N)], 0),
        )  # [..., P, N, N]
    else:
        fvar = Knn + tf.reduce_sum(
            T_Kun * (tf.matmul(K_tilde, T_Kun) - 2 * Kun), axis=-2
        )
        fvar = tf.broadcast_to(
            tf.expand_dims(fvar, -1),
            tf.concat([leading_dims, (N, num_func)], 0),
        )  # [..., N, P]

    return fmean, fvar


# NOTE: avoid tf.map_fn for performance reasons
@check_shapes(
    "Kun: [P, M, batch..., N]",
    "L_tilde: [P, M, M]",
    "Knn: [P, batch..., N, N] if full_cov",
    "Knn: [P, batch..., N] if not full_cov",
    "q_mu: [M, P]",
    "return[0]: [batch..., N, P]",  # TODO: R instead of P?
    "return[1]: [batch..., P, N, N] if full_cov",  # TODO: R instead of P?
    "return[1]: [batch..., N, P] if not full_cov",  # TODO: R instead of P?
)
def _base_likelihood_conditional_separate_independent(
    Kun: tf.Tensor,
    L_tilde: tf.Tensor,
    Knn: tf.Tensor,
    q_mu: tf.Tensor,
    *,
    full_cov: bool = False,
) -> MeanAndVariance:
    num_func = tf.shape(q_mu)[-1]
    M = tf.shape(q_mu)[-2]

    K = tf.rank(Kun)
    perm = tf.concat(
        [
            tf.range(2, K - 1),  # leading dims (...)
            (0, 1, K - 1),  # [P, M, N]
        ],
        0,
    )  # [N]
    Kun = tf.transpose(Kun, perm)  # [..., P, M, N]
    leading_dims = tf.shape(Kun)[:-3]

    L_tilde = tf.broadcast_to(
        L_tilde,
        tf.concat([leading_dims, (num_func, M, M)], 0),
    )  # [..., P, M, M]

    q_mu = tf.expand_dims(
        tf.broadcast_to(
            tf.transpose(q_mu),  # [P, M]
            tf.concat([leading_dims, (num_func, M)], 0),
        ),  # [..., P, M]
        axis=-1,
    )  # [..., P, M, 1]

    fmean = tf.matmul(Kun, q_mu, transpose_a=True)  # [..., P, N, 1]

    A = tf.linalg.triangular_solve(L_tilde, Kun)  # [..., P, M, N]

    if full_cov:
        K = tf.rank(Knn)
        perm = tf.concat(
            tf.range(1, K - 2),  # leading dims (...)
            (0, K - 2, K - 1),  # [P, N, N]
        )
        Knn = tf.transpose(Knn, perm)  # [..., P, N, N]

        fvar = Knn - tf.matmul(A, A, transpose_a=True)  # [..., P, N, N]
    else:
        K = tf.rank(Knn)
        perm = tf.concat(
            tf.range(1, K - 1),  # leading dims (...)
            (0, K - 1),  # [P, N]
        )
        Knn = tf.transpose(Knn, perm)  # [..., P, N]

        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., P, N]
        fvar = tf.linalg.matrix_transpose(fvar)  # [..., N, P]

    return fmean, fvar


# NOTE: avoid tf.map_fn for performance reasons
@check_shapes(
    "Kun: [P, M, batch..., N]",
    "K_tilde: [P, M, M]",
    "Knn: [P, batch..., N, N] if full_cov",
    "Knn: [P, batch..., N] if not full_cov",
    "q_mu: [M, P]",
    "q_T_sqrt: [M_M_or_P_M_M...]",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., P, N, N] if full_cov",
    "return[1]: [batch..., N, P] if not full_cov",
)
def _base_relaxed_conditional_separate_independent(
    Kun: tf.Tensor,
    K_tilde: tf.Tensor,
    Knn: tf.Tensor,
    q_mu: tf.Tensor,
    q_T_sqrt: tf.Tensor,
    *,
    full_cov: bool = False,
) -> MeanAndVariance:
    num_func = tf.shape(q_mu)[-1]
    M = tf.shape(q_mu)[-2]

    K = tf.rank(Kun)
    perm = tf.concat(
        [
            tf.range(2, K - 1),  # leading dims (...)
            (0, 1, K - 1),  # [P, M, N]
        ],
        0,
    )  # [N]
    Kun = tf.transpose(Kun, perm)  # [..., P, M, N]
    leading_dims = tf.shape(Kun)[:-3]

    K_tilde = tf.broadcast_to(
        K_tilde,
        tf.concat([leading_dims, (num_func, M, M)], 0),
    )  # [..., P, M, M]

    q_T_sqrt = tf.broadcast_to(
        q_T_sqrt,
        tf.concat([leading_dims, (num_func, M, M)], 0),
    )  # [..., P, M, M]

    q_mu = tf.expand_dims(
        tf.broadcast_to(
            tf.transpose(q_mu),  # [P, M]
            tf.concat([leading_dims, (num_func, M)], 0),
        ),  # [..., P, M]
        axis=-1,
    )  # [..., P, M, 1]

    fmean = tf.matmul(Kun, q_mu, transpose_a=True)  # [..., P, N, 1]

    # This is better if minibatch is small.
    T_Kun = tf.matmul(
        q_T_sqrt, tf.matmul(q_T_sqrt, Kun, transpose_a=True)
    )  # [..., P, M, N]

    # fvar = Knn + Kun^T * (T K_tilde T - 2T) Kun
    #   = Knn + Kun^T T (K_tilde T - 2I) Kun
    if full_cov:
        K = tf.rank(Knn)
        perm = tf.concat(
            tf.range(1, K - 2),  # leading dims (...)
            (0, K - 2, K - 1),  # [P, N, N]
        )
        Knn = tf.transpose(Knn, perm)  # [..., P, N, N]

        fvar = (
            Knn
            + tf.matmul(T_Kun, tf.matmul(K_tilde, T_Kun), transpose_a=True)
            - 2 * tf.matmul(Kun, T_Kun, transpose_a=True)
        )  # [..., P, N, N]
    else:
        K = tf.rank(Knn)
        perm = tf.concat(
            tf.range(1, K - 1),  # leading dims (...)
            (0, K - 1),  # [P, N]
        )
        Knn = tf.transpose(Knn, perm)  # [..., P, N]

        fvar = Knn + tf.reduce_sum(
            T_Kun * (tf.matmul(K_tilde, T_Kun) - 2 * Kun), axis=-2
        )  # [..., P, N]
        fvar = tf.linalg.matrix_transpose(fvar)  # [..., N, P]

    return fmean, fvar
