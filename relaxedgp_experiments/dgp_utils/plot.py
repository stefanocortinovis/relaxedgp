from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gpflow.base import RegressionData, default_float
from gpflux.models import DeepGP
from matplotlib.colors import ListedColormap


def _plot_model(
    model: DeepGP,
    *,
    task: str = "regression",
    data: Optional[RegressionData] = None,
    plot_kwargs: Optional[dict] = None,
) -> None:
    plot_kwargs = plot_kwargs or {}
    if task == "regression":
        return plot_1d_regression_model_dgp(model, data=data, **plot_kwargs)
    return plot_2d_classification_model(model, data=data, **plot_kwargs)


# TODO: propagate multiple samples through DGP
# TODO: plot inducing variables in first layer
def plot_1d_regression_model_dgp(
    model,
    data,
    *,
    xlim=None,
    plot_layers: bool = False,
) -> None:
    num_cols = 1 + len(model.f_layers) if plot_layers else 1
    fig, axs = plt.subplots(1, num_cols, figsize=(3 * num_cols, 3), squeeze=False)

    X, Y = data[0], data[1]

    ax = axs[0][0]
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")

    ax.plot(X, Y, "x")

    D = X.shape[1]
    data_inducing_pts = X

    assert D == 1

    if xlim is None:
        xlim = (
            np.min(data_inducing_pts) - 3.0,
            np.max(data_inducing_pts) + 3.0,
        )
    pX = np.linspace(
        xlim[0],
        xlim[1],
        300,
        dtype=default_float(),
    )[:, None]
    pred_model = model.as_prediction_model()
    out = pred_model(pX)
    pY = out.y_mean.numpy().squeeze()
    pYv = out.y_var.numpy().squeeze()

    (line,) = ax.plot(pX, pY, lw=1.5)
    col = line.get_color()
    ax.plot(pX, pY + 2 * pYv**0.5, col, lw=1.5)
    ax.plot(pX, pY - 2 * pYv**0.5, col, lw=1.5)

    inp = pX

    for i, (ax, layer) in enumerate(zip(axs[0][1:], model.f_layers)):
        ax.set_xlabel("$x$")
        ax.set_ylabel("$y$")
        ax.set_title(f"layer {i + 1}")

        gp_layer = layer.model
        out = gp_layer.predict_f(inp)[0]

        ax.plot(inp, out)

        inp = out
    return fig, ax


# TODO: implement for DGP
def plot_2d_classification_model(
    model: DeepGP,
    data: RegressionData,
    *,
    subsample: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots()

    D = 2

    cm_bright = ListedColormap(["#FF0000", "#0000FF"])

    X, Y = data[0], data[1]
    if subsample is not None and len(X) > subsample:
        idx = np.random.choice(len(X), subsample, replace=False)
        X, Y = X[idx], Y[idx]

    ax.scatter(
        X[:, 0],
        X[:, 1],
        c=Y,
        cmap=cm_bright,
        zorder=10,
        s=10,
    )

    data_inducing_pts = X

    x0_min, x0_max = np.min(data_inducing_pts[:, 0]), np.max(data_inducing_pts[:, 0])
    x1_min, x1_max = np.min(data_inducing_pts[:, 1]), np.max(data_inducing_pts[:, 1])

    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min - 1.0, x0_max + 1.0, 100),
        np.linspace(x1_min - 1.0, x1_max + 1.0, 100),
    )

    pX = np.stack((xx0, xx1), -1)
    pred_model = model.as_prediction_model()
    out = pred_model(pX.reshape(-1, D))
    pY = (out.y_mean.numpy() > 0.5).astype(int)
    pY = pY.reshape(xx0.shape)

    ax.contourf(
        xx0,
        xx1,
        pY,
        alpha=0.8,
        cmap=mpl.cm.RdBu,
        levels=np.linspace(0.0, 1.0, 10),
    )
    ax.contour(
        xx0,
        xx1,
        pY,
        levels=[0.5],
        colors="black",
        linestyles="dashed",
        linewidths=2,
        zorder=13,
    )

    return fig, ax
