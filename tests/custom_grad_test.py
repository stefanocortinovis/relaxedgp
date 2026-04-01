import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose


def test_custom_gradient():
    A = tf.convert_to_tensor(np.random.randn(3, 3))
    b = tf.convert_to_tensor(np.random.randn(3, 1))
    T = tf.linalg.inv(A)

    def f1(A, b):
        A_inv = tf.linalg.inv(A)
        return tf.linalg.matmul(A_inv, b)

    def f2(T, b):
        return tf.linalg.matmul(T, b)

    @tf.custom_gradient
    def f3(A, b, T):
        y = tf.matmul(T, b)

        def grad(upstream):
            dy_dA = -tf.matmul(
                tf.transpose(T),
                tf.matmul(upstream, tf.transpose(y)),
            )
            dy_db = tf.matmul(tf.transpose(T), upstream)
            dy_dT = tf.matmul(upstream, b, transpose_b=True)

            return dy_dA, dy_db, dy_dT

        return y, grad

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(A)
        tape.watch(b)
        tape.watch(T)

        y1 = f1(A, b)
        y2 = f2(T, b)
        y3 = f3(A, b, T)

    assert_allclose(y1, y2)
    assert_allclose(y1, y3)

    dy1_dA, dy1_db = tape.gradient(y1, [A, b])
    dy2_dT, dy2_db = tape.gradient(y2, [T, b])
    assert_allclose(dy1_db, dy2_db)

    dy3_dA, dy3_db, dy3_dT = tape.gradient(y3, [A, b, T])
    assert_allclose(dy1_dA, dy3_dA)
    assert_allclose(dy2_db, dy3_db)
    assert_allclose(dy2_dT, dy3_dT)
