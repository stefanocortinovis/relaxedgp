import os
from typing import Union

import gpflow
import gpflux
import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.keras import tf_keras
from gpflow.utilities import to_default_float

from relaxedgp.deep.layers import GPLayerWrapper


def load_data():
    path = os.path.join(os.path.dirname(__file__), "../data/snelson/data.csv.gz")
    data = np.loadtxt(path, delimiter=",", dtype=gpflow.default_float())
    return data[:, :-1], data[:, -1:]


def get_num_data(data):
    X, Y = data
    assert len(X) == len(Y)
    return len(X)


def make_dataset(data, as_dict=True):
    X, Y = data
    dataset_base = {"inputs": X, "targets": Y} if as_dict else (X, Y)
    batch_size = get_num_data(data)
    return tf.data.Dataset.from_tensor_slices(dataset_base).batch(batch_size)


def make_kernel_likelihood_iv():
    kernel = gpflow.kernels.SquaredExponential(variance=0.7, lengthscales=0.6)
    kernel.lengthscales.prior = tfp.distributions.LogNormal(
        to_default_float(1.0), to_default_float(0.5)
    )
    likelihood = gpflow.likelihoods.Gaussian(variance=0.08)
    Z = np.linspace(0, 6, 20)[:, np.newaxis]
    inducing_variable = gpflow.inducing_variables.InducingPoints(Z)
    gpflow.set_trainable(inducing_variable, False)
    return kernel, likelihood, inducing_variable


def create_gp_layer(kernel, inducing_variable, num_data):
    mok = gpflow.kernels.SharedIndependent(kernel, output_dim=1)
    moiv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducing_variable
    )
    model = gpflow.models.SVGP(
        mok, gpflow.likelihoods.Gaussian(), moiv, num_data=num_data
    )
    gpflow.set_trainable(model.likelihood, False)
    return GPLayerWrapper(model)


def create_gp_layer_gpflux(kernel, inducing_variable, num_data):
    mok = gpflow.kernels.SharedIndependent(kernel, output_dim=1)
    moiv = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducing_variable
    )
    return gpflux.layers.GPLayer(
        mok, moiv, num_data, mean_function=gpflow.mean_functions.Zero()
    )


def create_sldgp(kernel, likelihood, inducing_variable, num_data):
    gp_layer = create_gp_layer(kernel, inducing_variable, num_data)
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([gp_layer], likelihood_layer, num_data=num_data)
    return model


def create_sldgp_gpflux(kernel, likelihood, inducing_variable, num_data):
    gp_layer = create_gp_layer_gpflux(kernel, inducing_variable, num_data)
    likelihood_layer = gpflux.layers.LikelihoodLayer(likelihood)
    model = gpflux.models.DeepGP([gp_layer], likelihood_layer, num_data=num_data)
    return model


def assign_to_gpflux(sldgp, sldgp_gpflux):
    [gp_layer] = sldgp.f_layers
    [gp_layer_gpflux] = sldgp_gpflux.f_layers
    gp_layer_gpflux.q_mu.assign(gp_layer.model.q_mu)
    gp_layer_gpflux.q_sqrt.assign(gp_layer.model.q_sqrt)
    kernel = gp_layer_gpflux.kernel.kernel
    kernel.variance.assign(gp_layer.model.kernel.kernel.variance)
    kernel.lengthscales.assign(gp_layer.model.kernel.kernel.lengthscales)
    iv = gp_layer_gpflux.inducing_variable.inducing_variable
    iv.Z.assign(gp_layer.model.inducing_variable.inducing_variable.Z)
    sldgp_gpflux.likelihood_layer.likelihood.variance.assign(
        sldgp.likelihood_layer.likelihood.variance
    )


def fit_scipy(model, data, maxiter=100):
    def training_loss():
        return -model.elbo(data)

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        training_loss, model.trainable_variables, options=dict(maxiter=maxiter)
    )


def fit_adam(
    model: Union[gpflow.models.SVGP, gpflux.models.DeepGP],
    data,
    maxiter,
    adam_learning_rate=0.01,
):
    X, Y = data
    num_data = len(X)

    def training_loss():
        """
        NOTE: the Keras model.compile()/fit() uses the implicit losses, which are
        computed as

        >>> _ = model(data, training=True)
        >>> return tf.reduce_sum(model.losses)

        The scaling factor leads to a O(1e-3) discrepancy between approaches; to have
        an exact comparison we therefore re-scale the objective here.
        """
        return -model.elbo(data) / num_data

    adam = tf_keras.optimizers.Adam(adam_learning_rate)

    @tf.function
    def optimization_step():
        adam.minimize(training_loss, var_list=model.trainable_variables)

    for i in range(maxiter):
        optimization_step()


def _keras_fit_adam(model, dataset, maxiter, adam_learning_rate=0.01, loss=None):
    model.compile(optimizer=tf_keras.optimizers.Adam(adam_learning_rate), loss=loss)
    model.fit(dataset, epochs=maxiter)


def keras_fit_adam(sldgp: gpflux.models.DeepGP, data, maxiter, adam_learning_rate=0.01):
    model = sldgp.as_training_model()
    dataset = make_dataset(data)
    _keras_fit_adam(model, dataset, maxiter, adam_learning_rate=adam_learning_rate)


def run_sldgp_gpflux(data, optimizer, maxiter):
    kernel, likelihood, inducing_variable = make_kernel_likelihood_iv()
    num_data = len(data[0])
    model = create_sldgp_gpflux(kernel, likelihood, inducing_variable, num_data)
    if optimizer == "adam":
        fit_adam(model, data, maxiter=maxiter)
    elif optimizer == "scipy":
        pytest.skip("Numerically unstable")
        fit_scipy(model, data, maxiter=maxiter)
    elif optimizer == "keras_adam":
        keras_fit_adam(model, data, maxiter=maxiter)
    else:
        raise NotImplementedError
    return model


def run_sldgp(data, optimizer, maxiter):
    kernel, likelihood, inducing_variable = make_kernel_likelihood_iv()
    num_data = len(data[0])
    model = create_sldgp(kernel, likelihood, inducing_variable, num_data)
    if optimizer == "adam":
        fit_adam(model, data, maxiter=maxiter)
    elif optimizer == "scipy":
        fit_scipy(model, data, maxiter=maxiter)
    elif optimizer == "keras_adam":
        keras_fit_adam(model, data, maxiter=maxiter)
    else:
        raise NotImplementedError
    return model


def assert_equivalence(sldgp, sldgp_gpflux, data, **tol_kws):
    X, Y = data
    np.testing.assert_allclose(sldgp.elbo(data), sldgp_gpflux.elbo(data), **tol_kws)
    np.testing.assert_allclose(sldgp.predict_f(X), sldgp_gpflux.predict_f(X), **tol_kws)


def test_svgp_equivalence_after_assign():
    data = load_data()
    sldgp = create_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    fit_scipy(sldgp, data)  # TODO: numerically unstable
    sldgp_gpflux = create_sldgp_gpflux(*make_kernel_likelihood_iv(), get_num_data(data))
    assign_to_gpflux(sldgp, sldgp_gpflux)
    assert_equivalence(sldgp, sldgp_gpflux, data)


@pytest.mark.parametrize(
    "fitter, tol_kw",
    [
        (fit_adam, {}),
        (keras_fit_adam, {}),
    ],
)
def test_svgp_equivalence_with_sldgp(fitter, tol_kw, maxiter=20):
    data = load_data()

    sldgp = create_sldgp(*make_kernel_likelihood_iv(), get_num_data(data))
    fitter(sldgp, data, maxiter=maxiter)

    sldgp_gpflux = create_sldgp_gpflux(*make_kernel_likelihood_iv(), get_num_data(data))
    fitter(sldgp_gpflux, data, maxiter=maxiter)

    assert_equivalence(sldgp, sldgp_gpflux, data, **tol_kw)


@pytest.mark.parametrize(
    "optimizer",
    ["adam", "scipy", "keras_adam"],
)
def test_run_sldgp_gpflux(optimizer):
    data = load_data()
    _ = run_sldgp_gpflux(data, optimizer, maxiter=10)


@pytest.mark.parametrize("optimizer", ["adam", "scipy", "keras_adam"])
def test_run_sldgp(optimizer):
    data = load_data()
    _ = run_sldgp(data, optimizer, maxiter=10)
