from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow.base import MeanAndVariance, RegressionData
from gpflux.models import DeepGP


@tf.function(jit_compile=True, reduce_retracing=True)
def _training_loss(
    model: DeepGP,
    data: RegressionData,
) -> tf.Tensor:
    return -model.elbo(data)


def training_loss(
    model: DeepGP,
    data: Optional[RegressionData] = None,
    batch_size: Optional[int] = None,
) -> tf.Tensor:
    if batch_size is None:
        return _training_loss(model, data)

    X, Y = data
    dataset_size = len(X)
    num_batches = dataset_size // batch_size

    loss = 0.0
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size

        X_batch = X[start:end]
        Y_batch = Y[start:end]

        loss += _training_loss(model, (X_batch, Y_batch))

    return loss / num_batches


@tf.function(jit_compile=True, reduce_retracing=True)
def predict_f(
    model: DeepGP,
    X: RegressionData,
    num_samples: int = 1,
) -> MeanAndVariance:
    num_samples_first_layer = model.f_layers[0].num_samples
    if num_samples_first_layer is not None:
        num_samples = num_samples // num_samples_first_layer
        num_samples_total = num_samples_first_layer * num_samples
    else:
        num_samples_total = num_samples

    N = tf.shape(X)[0]
    pred_model = model.as_prediction_model()
    X_samples = tf.tile(X, (num_samples, 1))
    pred = pred_model(X_samples)
    pred_mean, pred_var = pred.f_mean, pred.f_var

    P = pred_mean.shape[-1]
    pred_mean = tf.reshape(pred_mean, (num_samples_total, N, P))
    pred_var = tf.reshape(pred_var, (num_samples_total, N, P))
    return pred_mean, pred_var


@tf.function(jit_compile=True, reduce_retracing=True)
def predict_log_density(
    model: DeepGP, data: RegressionData, pred: MeanAndVariance, task: str = "regression"
) -> tf.Tensor:
    X, Y = data  # [N, D], [N, P]
    pred_mean, pred_var = pred  # [S, N, P], [S, N, P]

    if "classification_multi" in task:
        log_density = tf.vectorized_map(
            lambda z: model.likelihood_layer.likelihood.predict_log_density(
                X, z[0], z[1], Y
            ),
            (pred_mean, pred_var),
        )
    else:
        log_density = model.likelihood_layer.likelihood.predict_log_density(
            X, pred_mean, pred_var, Y
        )
    return tf.reduce_mean(log_density, axis=0)


def evaluation_metrics(
    model: DeepGP,
    data: RegressionData,
    batch_size: Optional[int] = None,
    task: str = "regression",
    num_samples: int = 1,
):
    X, Y = data

    dataset_size = len(X)
    if batch_size is None:
        batch_size = dataset_size

    error = 0.0
    lpd = 0.0
    for start in range(0, dataset_size, batch_size):
        end = start + batch_size

        X_batch = X[start:end]
        Y_batch = Y[start:end]

        pred_mean, pred_var = predict_f(model, X_batch, num_samples)

        if task == "regression":
            error += np.sum(np.mean((Y_batch - pred_mean) ** 2, axis=0))
        elif task == "classification":
            pred_mean = tf.reduce_mean(
                model.likelihood_layer.likelihood.predict_mean_and_var(
                    X_batch, pred_mean, pred_var
                )[0],  # [S, N, P]
                axis=0,
            )  # [N, P]
            error += np.sum(Y_batch != tf.cast(pred_mean > 0.5, Y_batch.dtype))
        elif "classification_multi" in task:
            pred_mean_ = tf.reduce_mean(pred_mean, axis=0)
            error += np.sum(Y_batch[:, 0] != tf.argmax(pred_mean_, axis=-1))

        lpd += np.sum(
            predict_log_density(model, (X_batch, Y_batch), (pred_mean, pred_var), task)
        )

    error /= dataset_size
    lpd /= dataset_size

    if task == "regression":
        error = np.sqrt(error)

    return {
        "test/error": error,  # rmse for regression, error for classification
        "test/nlpd": -lpd,
    }
