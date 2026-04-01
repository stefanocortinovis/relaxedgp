from typing import Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from gpflow.base import RegressionData, default_float
from matplotlib.colors import ListedColormap

from relaxedgp.typing import Model


def _plot_model(
    model: Model,
    *,
    task: str = "regression",
    data: Optional[RegressionData] = None,
    plot_kwargs: Optional[dict] = None,
) -> None:
    plot_kwargs = plot_kwargs or {}
    if task == "regression":
        return plot_1d_regression_model(model, data=data, **plot_kwargs)
    return plot_2d_classification_model(model, data=data, **plot_kwargs)


def plot_1d_regression_model(
    model: Model,
    *,
    data: Optional[RegressionData] = None,
    xlim: Optional[Tuple[float, float]] = None,
) -> None:
    fig, ax = plt.subplots()

    if data is None and not hasattr(model, "inducing_variable"):
        raise ValueError("At least one of data and inducing_variable must be provided.")

    if data is not None:
        X, Y = data[0], data[1]
        ax.plot(X, Y, "x")

        D = X.shape[1]
        data_inducing_pts = X
    else:
        D = model.inducing_variable.Z.numpy().shape[1]
        data_inducing_pts = np.empty((0, D))

    assert D == 1

    if hasattr(model, "inducing_variable"):
        inducing_pts = model.inducing_variable.Z.numpy()
        data_inducing_pts = np.vstack([data_inducing_pts, inducing_pts])

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
    pY, pYv = model.predict_y(pX)

    (line,) = ax.plot(pX, pY, lw=1.5)
    col = line.get_color()
    ax.plot(pX, pY + 2 * pYv**0.5, col, lw=1.5)
    ax.plot(pX, pY - 2 * pYv**0.5, col, lw=1.5)

    if hasattr(model, "inducing_variable"):
        ax.plot(
            inducing_pts,
            np.broadcast_to(ax.get_ylim()[0], inducing_pts.shape),
            "k|",
            mew=2,
        )

    return fig, ax


def plot_2d_classification_model(
    model: Model,
    *,
    data: Optional[RegressionData] = None,
    subsample: Optional[int] = None,
) -> None:
    fig, ax = plt.subplots()

    D = 2

    if data is None and not hasattr(model, "inducing_variable"):
        raise ValueError("At least one of data and inducing_variable must be provided.")

    if data is not None:
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
    else:
        data_inducing_pts = np.empty((0, D))

    if hasattr(model, "inducing_variable"):
        inducing_pts = model.inducing_variable.Z.numpy()
        data_inducing_pts = np.vstack([data_inducing_pts, inducing_pts])

        ax.scatter(
            inducing_pts[:, 0],
            inducing_pts[:, 1],
            color="black",
            marker="x",
            zorder=11,
        )

    x0_min, x0_max = np.min(data_inducing_pts[:, 0]), np.max(data_inducing_pts[:, 0])
    x1_min, x1_max = np.min(data_inducing_pts[:, 1]), np.max(data_inducing_pts[:, 1])

    xx0, xx1 = np.meshgrid(
        np.linspace(x0_min - 1.0, x0_max + 1.0, 100),
        np.linspace(x1_min - 1.0, x1_max + 1.0, 100),
    )

    pX = np.stack((xx0, xx1), -1)
    pY, _ = model.predict_y(pX.reshape(-1, D))
    pY = pY.numpy().reshape(xx0.shape)

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
