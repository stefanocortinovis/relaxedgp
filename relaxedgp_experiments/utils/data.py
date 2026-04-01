from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import openml
import tensorflow as tf
import uci_datasets
from gpflow.base import RegressionData, TensorType, default_float
from sklearn.preprocessing import LabelEncoder


@dataclass
class Dataset:
    name: str
    train_data: RegressionData
    test_data: Optional[RegressionData]
    stats: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]

    def __init__(
        self,
        name: str,
        split: int = 0,
        train_only: bool = False,
        origin: str = "uci",
        normalize: bool = True,
    ) -> None:
        self.name = f"{name}_{split}"

        train_data, test_data = self.load_data(
            name,
            split,
            train_only,
            origin,
        )

        self.stats = _get_stats(train_data) if normalize else None
        self.train_data = self.preprocess_data(train_data)
        if test_data is not None:
            test_data = self.preprocess_data(test_data)
        self.test_data = test_data

    def load_data(
        self,
        name: str,
        split: int = 0,
        train_only: bool = False,
        origin: str = "uci",
    ) -> Tuple[RegressionData, Optional[RegressionData]]:
        if origin == "uci":
            data_raw = uci_datasets.Dataset(
                name,
                dtype=default_float(),
                print_stats=False,
            ).get_split(split)
            if train_only:
                x_train, y_train, x_test, y_test = data_raw
                x_train = np.concatenate([x_train, x_test], axis=0)
                y_train = np.concatenate([y_train, y_test], axis=0)
                return (x_train, y_train), None

            return data_raw[:2], data_raw[2:]
        elif origin == "openml":
            data_raw = (
                openml.datasets.get_dataset(
                    name,
                    download_data=False,
                    download_qualities=False,
                    download_features_meta_data=False,
                )
                .get_data()[0]
                .to_numpy()
            )
            if train_only:
                X_raw = data_raw[:, :-1].astype(default_float())
                Y_raw = (
                    LabelEncoder()
                    .fit_transform(data_raw[:, -1])
                    .reshape(-1, 1)
                    .astype(default_float())
                )
                return (X_raw, Y_raw), None

            test_len = int(data_raw.shape[0] * 0.1)
            perm = np.random.RandomState(split).permutation(data_raw.shape[0])
            test_data_raw = data_raw[perm[:test_len]]
            train_data_raw = data_raw[perm[test_len:]]

            encoder = LabelEncoder()
            X_train_raw = train_data_raw[:, :-1].astype(default_float())
            Y_train_raw = (
                encoder.fit_transform(train_data_raw[:, -1])
                .reshape(-1, 1)
                .astype(default_float())
            )
            X_test_raw = test_data_raw[:, :-1].astype(default_float())
            Y_test_raw = (
                encoder.transform(test_data_raw[:, -1])
                .reshape(-1, 1)
                .astype(default_float())
            )
            return (X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw)
        elif origin == "local":
            train_data_raw = np.loadtxt(
                fname=Path(__file__).parent.parent.parent
                / "data"
                / name
                / "data.csv.gz",
                dtype=default_float(),
                delimiter=",",
            )
            if train_only:
                train_data_raw = (train_data_raw[:, :-1], train_data_raw[:, -1:])
                return train_data_raw, None

            test_len = int(train_data_raw.shape[0] * 0.1)
            perm = np.random.RandomState(split).permutation(train_data_raw.shape[0])
            test_data_raw = train_data_raw[perm[:test_len]]
            train_data_raw = train_data_raw[perm[test_len:]]

            return (train_data_raw[:, :-1], train_data_raw[:, -1:]), (
                test_data_raw[:, :-1],
                test_data_raw[:, -1:],
            )
        elif name == "mnist_784":
            assert not train_only

            (
                (X_train_raw, Y_train_raw),
                (X_test_raw, Y_test_raw),
            ) = tf.keras.datasets.mnist.load_data()

            encoder = LabelEncoder()

            X_train_raw = X_train_raw.reshape(-1, 28 * 28).astype(default_float())
            Y_train_raw = (
                encoder.fit_transform(Y_train_raw)
                .reshape(-1, 1)
                .astype(default_float())
            )

            X_test_raw = X_test_raw.reshape(-1, 28 * 28).astype(default_float())
            Y_test_raw = (
                encoder.transform(Y_test_raw).reshape(-1, 1).astype(default_float())
            )

            return (X_train_raw, Y_train_raw), (X_test_raw, Y_test_raw)
        else:
            raise ValueError(f"Unknown origin: {origin}")

    def preprocess_data(
        self,
        data: RegressionData,
    ) -> Tuple[TensorType, TensorType]:
        if self.stats is None:
            return data

        X, Y = data

        if "mnist_784" in self.name:
            # For MNIST, center the data in [0, 1]
            # X = X / 255.0 * 2 - 1.0
            X = X / 255.0
            return X, Y

        X_mean, X_std, Y_mean, Y_std = self.stats
        return (X - X_mean) / X_std, (Y - Y_mean) / Y_std


def _get_stats(
    data: RegressionData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X, Y = data
    return (
        np.mean(X, axis=0),
        np.std(X, axis=0) + 1e-6,
        np.mean(Y, axis=0),
        np.std(Y, axis=0) + 1e-6,
    )
