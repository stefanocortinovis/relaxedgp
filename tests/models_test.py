import gpflow
import numpy as np
from gpflow.models import SVGP

from relaxedgp.models import LSVGP, RSVGP
from relaxedgp_experiments.utils.data import Dataset


def get_trained_snelson_model(model_class=LSVGP):
    d = Dataset("snelson", train_only=True, origin="local", normalize=False)
    X, Y = d.train_data

    lik = model_class(
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.Gaussian(),
        np.linspace(0, 6, 10)[:, None],
        num_data=len(X),
    )

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        lambda: lik.training_loss((X, Y)),
        lik.trainable_variables,
        options=dict(maxiter=10000),
    )

    return lik


def test_elbo_equality():
    d = Dataset("snelson", train_only=True, origin="local", normalize=False)
    X, Y = d.train_data
    lik = get_trained_snelson_model()

    q_mu, q_var = lik.predict_f(lik.inducing_variable.Z, full_cov=True)

    svgp = SVGP(
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.Gaussian(),
        lik.inducing_variable.Z.numpy(),
        num_data=len(X),
        q_mu=q_mu,
        q_sqrt=np.linalg.cholesky(q_var[None, :, :]),
        whiten=False,
    )
    params = gpflow.utilities.read_values(lik)
    del params[".q_mu"]
    del params[".q_sqrt"]
    gpflow.utilities.multiple_assign(svgp, params)

    lik_elbo, svgp_elbo = [m.elbo((X, Y)).numpy() for m in [lik, svgp]]
    pd = np.abs((lik_elbo - svgp_elbo) / svgp_elbo) * 100
    assert pd < 0.01

    rsvgp = RSVGP(
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.Gaussian(),
        lik.inducing_variable.Z.numpy(),
        num_data=len(X),
    )
    gpflow.utilities.multiple_assign(rsvgp, gpflow.utilities.read_values(lik))

    elbo_r_suboptimal_T = rsvgp.elbo((X, Y))
    K_tilde = rsvgp.K_tilde()
    L_tilde = np.linalg.cholesky(np.linalg.inv(K_tilde))
    rsvgp.q_T_sqrt.assign(L_tilde[None])

    elbo_r_optimal_T = rsvgp.elbo((X, Y))

    pd = np.abs((elbo_r_optimal_T - lik_elbo) / lik_elbo) * 100
    assert pd < 1e-6
    assert elbo_r_suboptimal_T < elbo_r_optimal_T
