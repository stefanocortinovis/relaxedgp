from typing import Optional, Union

import numpy as np
import tensorflow as tf
from gpflow.base import MeanAndVariance, RegressionData
from gpflow.models import (
    GPR,
    SGPR,
    ExternalDataTrainingLossMixin,
    InternalDataTrainingLossMixin,
)

from relaxedgp.typing import Model


@tf.function(jit_compile=True, reduce_retracing=True)
def _training_loss(
    model: Union[ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin],
    data: Optional[RegressionData] = None,
) -> tf.Tensor:
    if isinstance(model, (GPR, SGPR)):
        return model.training_loss()
    return model.training_loss(data)


def training_loss(
    model: Union[ExternalDataTrainingLossMixin, InternalDataTrainingLossMixin],
    data: Optional[RegressionData] = None,
    batch_size: Optional[int] = None,
) -> tf.Tensor:
    if batch_size is None:
        return _training_loss(model, data)
    elif isinstance(model, (GPR, SGPR)):
        raise ValueError("Batch size is not applicable for GPR or SGPR models.")

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
    model: Model,
    data: RegressionData,
) -> MeanAndVariance:
    return model.predict_f(data)


@tf.function(jit_compile=True, reduce_retracing=True)
def predict_log_density(
    model: Model, data: RegressionData, pred: MeanAndVariance
) -> tf.Tensor:
    X, Y = data
    pred_mean, pred_var = pred
    return model.likelihood.predict_log_density(X, pred_mean, pred_var, Y)


def evaluation_metrics(
    model: Model,
    data: RegressionData,
    batch_size: Optional[int] = None,
    task: str = "regression",
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

        pred_mean, pred_var = predict_f(model, X_batch)

        if task == "regression":
            error += np.sum((Y_batch - pred_mean) ** 2)
        elif task == "classification":
            error += np.sum(Y_batch != tf.cast(pred_mean > 0, Y_batch.dtype))
        elif "classification_multi" in task:
            error += np.sum(Y_batch[:, 0] != tf.argmax(pred_mean, axis=-1))

        lpd += np.sum(
            predict_log_density(model, (X_batch, Y_batch), (pred_mean, pred_var))
        )

    error /= dataset_size
    lpd /= dataset_size

    if task == "regression":
        error = np.sqrt(error)

    return {
        "test/error": error,  # rmse for regression, error for classification
        "test/nlpd": -lpd,
    }
