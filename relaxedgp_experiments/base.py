import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import hydra
import numpy as np
import tensorflow as tf
import wandb
from gpflow.base import InputData, OutputData, Parameter, RegressionData
from gpflow.config import default_jitter
from gpflow.inducing_variables.inducing_patch import InducingPatches
from gpflow.kernels import ArcCosine, Convolutional, Kernel, SquaredExponential
from gpflow.likelihoods import Bernoulli, Gaussian, Likelihood, MultiClass, RobustMax
from gpflow.models import GPR, SGPR, SVGP
from gpflow.utilities import set_trainable
from omegaconf import DictConfig

from relaxedgp.covariances import Kuf, Kuu  # noqa: F401
from relaxedgp.inducing_variables import (
    ConditionalVariance,
    Kmeans,
    UniformSubsample,
)
from relaxedgp.kernels import GammaExponential, WhitenedConvolutional
from relaxedgp.models import LSVGP, RSVGP
from relaxedgp.schedules import NaturalGradientSchedule
from relaxedgp.typing import Model
from relaxedgp_experiments.utils.data import Dataset
from relaxedgp_experiments.utils.evaluate import evaluation_metrics, training_loss
from relaxedgp_experiments.utils.log import get_callback
from relaxedgp_experiments.utils.plot import _plot_model
from relaxedgp_experiments.utils.train import train


@dataclass
class Experiment:
    log_dir: str
    task: str
    train_data: RegressionData
    test_data: Optional[RegressionData]
    model: Model
    logger: logging.Logger

    def __init__(
        self,
        log_dir: str,
        dataset_name: str,
        model_name: str,
        task: str = "regression",
        kernel_name: str = "squared_exponential",
        train_only: bool = False,
        origin: str = "uci",
        normalize_data: bool = False,
        M: Optional[int] = None,
        init_Z_method: str = "random",
        preconditioner: str = "identity",
        logger: Optional[logging.Logger] = None,
        debug_initialise_optimal_T: bool = False,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        eval_batch_size: Optional[int] = None,
        num_probes: Optional[int] = None,
    ) -> None:
        self.logger = logger or logging.getLogger("dummy")
        self.logger.info("Setting up experiment...")

        self.log_dir = log_dir
        self.task = task
        self.train_data, self.test_data = self._load_data(
            dataset_name, train_only, origin, normalize_data
        )
        self.model = self._setup_model(
            model_name,
            M,
            task,
            kernel_name,
            init_Z_method,
            preconditioner,
            debug_initialise_optimal_T=debug_initialise_optimal_T,
            num_probes=num_probes,
        )
        self.plot_kwargs = plot_kwargs or dict()
        self.eval_batch_size = eval_batch_size

    def run(self, **kwargs: Any) -> List[float]:
        self.logger.info("Running experiment...")
        lmls = self.train_model(**kwargs)
        self.save_model()
        if kwargs.get("verbose", True) and (
            (self.D == 1 and self.task == "regression")
            or (self.D == 2 and self.task == "classification")
        ):
            self.plot_model()
        self.logger.info("Experiment finished.")
        return lmls

    def evaluate_model(self) -> Dict:
        if self.test_data is None:
            return dict()
        return evaluation_metrics(
            self.model,
            self.test_data,
            batch_size=self.eval_batch_size,
            task=self.task,
        )

    def plot_model(self) -> None:
        save_dir = os.path.join(self.log_dir, "figures")
        self.logger.info(f"Plotting model in {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        fig, ax = _plot_model(
            self.model,
            task=self.task,
            data=self.train_data,
            plot_kwargs=self.plot_kwargs,
        )
        fig.savefig(os.path.join(save_dir, f"model_{self.task}.png"))

    def save_model(self) -> None:
        save_dir = os.path.join(self.log_dir, "checkpoints")
        self.logger.info(f"Saving model in {save_dir}...")
        os.makedirs(save_dir, exist_ok=True)
        ckpt = tf.train.Checkpoint(model=self.model)
        manager = tf.train.CheckpointManager(ckpt, save_dir, max_to_keep=3)
        manager.save()

    def train_model(
        self,
        n_iter: int = 10000,
        adam_lr: float = 1e-3,
        natgrad_schedule: Optional[NaturalGradientSchedule] = None,
        train_Z: bool = False,
        batch_size: Optional[int] = None,
        evaluate_every: Optional[int] = None,
        log_wandb: bool = False,
        log_quality_T: bool = False,
        reduce_on_plateau: bool = False,
        reduce_on_plateau_Z: bool = False,
        debug_force_bfgs: bool = False,
        debug_full_loss: bool = False,
        debug_log_all_T: bool = False,
        warmup_Z: int = 0,
        adam_lr_Z: Optional[float] = None,
        adam_beta_1_Z: float = 0.9,
        debug_grad_Z: bool = False,
        debug_movement_Z: bool = False,
        **kwargs: Any,
    ) -> List[float]:
        self.logger.info("Training model...")

        if evaluate_every is None:
            evaluate_every = n_iter

        callback = get_callback(
            self.model,
            evaluate_func=self.evaluate_model,
            evaluate_every=evaluate_every,
            logger=self.logger,
            log_wandb=log_wandb,
            log_quality_T=log_quality_T,
            n_iter=n_iter,
            debug_force_bfgs=debug_force_bfgs,
            debug_log_all_T=debug_log_all_T,
        )

        start = time.time()
        lmls = train(
            self.model,
            self.train_data,
            n_iter=n_iter,
            adam_lr=adam_lr,
            natgrad_schedule=natgrad_schedule,
            train_Z=train_Z,
            batch_size=batch_size,
            logger=self.logger,
            log_wandb=log_wandb,
            callback=callback,
            reduce_on_plateau=reduce_on_plateau,
            reduce_on_plateau_Z=reduce_on_plateau_Z,
            debug_force_bfgs=debug_force_bfgs,
            debug_full_loss=debug_full_loss,
            warmup_Z=warmup_Z,
            adam_lr_Z=adam_lr_Z,
            adam_beta_1_Z=adam_beta_1_Z,
            debug_grad_Z=debug_grad_Z,
            debug_movement_Z=debug_movement_Z,
            **kwargs,
        )
        end = time.time()
        exec_time = end - start
        self.logger.info(f"Execution time: {exec_time:.2f} seconds.")

        final_loss = (
            training_loss(self.model, self.train_data, batch_size=batch_size)
            .numpy()
            .item()
        )
        self.logger.info(f"Final training loss: {final_loss:.2f}.")

        if (run := wandb.run) is not None:
            run.log({"train/loss_final": final_loss})

            final_metrics = self.evaluate_model()
            for key, value in final_metrics.items():
                run.log({f"{key}_final": value})
            run.finish()

        return lmls

    @staticmethod
    def setup_from_config(
        cfg: DictConfig, log_dir: str, logger: Optional[logging.Logger] = None
    ) -> "Experiment":
        num_probes = cfg.model.get("num_probes", None)
        if num_probes is not None and num_probes > cfg.model.M:
            num_probes = None

        return Experiment(
            log_dir=log_dir,
            dataset_name=cfg.dataset.name,
            model_name=cfg.model.name,
            task=cfg.dataset.task,
            kernel_name=cfg.model.kernel_name,
            train_only=cfg.dataset.train_only,
            origin=cfg.dataset.origin,
            normalize_data=cfg.dataset.normalize,
            M=cfg.model.M,
            init_Z_method=cfg.model.init_Z_method,
            preconditioner=cfg.model.preconditioner,
            logger=logger,
            debug_initialise_optimal_T=cfg.debug.initialise_optimal_T,
            eval_batch_size=cfg.eval.batch_size,
            num_probes=num_probes,
        )

    def run_from_config(self, cfg: DictConfig) -> None:
        natgrad_schedule = hydra.utils.instantiate(cfg.training.natgrad_schedule)
        if natgrad_schedule.stopping_criterion is not None:
            natgrad_schedule.stopping_criterion.num_data = self.N
        self.run(
            n_iter=cfg.training.n_iter,
            adam_lr=cfg.training.adam_lr,
            natgrad_schedule=natgrad_schedule,
            train_Z=cfg.training.train_Z,
            batch_size=cfg.training.batch_size,
            evaluate_every=cfg.eval.evaluate_every,
            log_wandb=cfg.eval.log_wandb,
            log_quality_T=cfg.eval.log_quality_T,
            reduce_on_plateau=cfg.training.reduce_on_plateau,
            reduce_on_plateau_Z=cfg.training.reduce_on_plateau_Z,
            debug_force_bfgs=cfg.debug.force_bfgs,
            debug_full_loss=cfg.debug.full_loss,
            debug_log_all_T=cfg.debug.log_all_T,
            warmup_Z=cfg.training.warmup_Z,
            adam_lr_Z=cfg.training.adam_lr_Z,
            adam_beta_1_Z=cfg.training.adam_beta_1_Z,
            debug_grad_Z=cfg.debug.grad_Z,
            debug_movement_Z=cfg.debug.movement_Z,
        )

    @property
    def X_train(self) -> InputData:
        return self.train_data[0]

    @property
    def Y_train(self) -> OutputData:
        return self.train_data[1]

    @property
    def N(self) -> int:
        return len(self.X_train)

    @property
    def D(self) -> int:
        return self.X_train.shape[1]

    @staticmethod
    def _load_data(
        dataset_name: str,
        train_only: bool = False,
        origin: str = "uci",
        normalize_data: bool = False,
    ) -> RegressionData:
        data = Dataset(
            dataset_name,
            train_only=train_only,
            origin=origin,
            normalize=normalize_data,
        )
        return data.train_data, data.test_data

    def _setup_likelihood(self, task: str = "regression") -> Likelihood:
        if task == "regression":
            return Gaussian(), 1
        elif task == "classification":
            return Bernoulli(), 1
        elif "classification_multi" in task:
            _, _, P = task.split("_")
            P = int(P)
            invlink = RobustMax(P)
            return MultiClass(P, invlink), P
        else:
            raise ValueError(f"Unknown task {task}")

    def _setup_kernel(self, kernel_name: str, model_name: str, **kwargs: Any) -> Kernel:
        if kernel_name == "squared_exponential":
            return SquaredExponential(
                variance=1.0,
                lengthscales=np.ones(self.D),
                name=model_name,
            )
        elif kernel_name == "gamma_exponential":
            return GammaExponential(
                variance=1.0,
                lengthscales=np.ones(self.D),
                gamma=1.1,
                name=model_name,
            )
        elif kernel_name == "arc_cosine_0":
            return ArcCosine(
                order=0,
                variance=1.0,
                weight_variances=np.ones(self.D),
                bias_variance=1.0,
            )
        elif kernel_name == "arc_cosine_1":
            return ArcCosine(
                order=1,
                variance=1.0,
                weight_variances=np.ones(self.D),
                bias_variance=1.0,
            )
        elif "conv" in kernel_name:
            image_shape = kwargs.get("image_shape")
            patch_shape = kwargs.get("patch_shape", (5, 5))

            convolutional_kernel = Convolutional(
                base_kernel=SquaredExponential(),
                image_shape=image_shape,
                patch_shape=patch_shape,
            )

            if kernel_name == "conv":
                return convolutional_kernel
            elif kernel_name == "conv_whitened":
                return WhitenedConvolutional(
                    convolutional_kernel=convolutional_kernel,
                    white_variance=1e-3,
                )
        else:
            raise ValueError(f"Unknown kernel class {kernel_name}")

    def _setup_model(
        self,
        model_name: str,
        M: Optional[int],
        task: str = "regression",
        kernel_name: str = "squared_exponential",
        init_Z_method: str = "random",
        preconditioner: str = "identity",
        debug_initialise_optimal_T: bool = False,
        num_probes: Optional[int] = None,
    ) -> Model:
        if "conv" in kernel_name:
            image_width = int(np.sqrt(self.X_train.shape[1]))
            kwargs = {"image_shape": (image_width, image_width)}
        else:
            kwargs = {}

        kernel = self._setup_kernel(kernel_name, model_name, **kwargs)

        if model_name == "gpr":
            model = GPR(self.train_data, kernel)
        else:
            N = self.N
            if M is None:
                M = N

            if init_Z_method == "random":
                init_Z_method = UniformSubsample(seed=0)
            elif init_Z_method == "conditional_variance":
                init_Z_method = ConditionalVariance()
            elif init_Z_method == "kmeans":
                init_Z_method = Kmeans()
            else:
                raise ValueError(f"Unknown Z initialization method {init_Z_method}")

            Z = init_Z_method(self.X_train, M, kernel)[0].copy()
            if "conv" in kernel_name:
                if kernel_name == "conv":
                    Z = (
                        kernel.get_patches(Z)
                        .numpy()
                        .reshape(
                            -1,
                            np.prod(kernel.patch_shape),
                        )
                    )
                elif kernel_name == "conv_whitened":
                    Z = (
                        kernel.convolutional_kernel.get_patches(Z)
                        .numpy()
                        .reshape(
                            -1,
                            np.prod(kernel.convolutional_kernel.patch_shape),
                        )
                    )

                Z = Z[np.random.permutation(len(Z))[:M]]
                Z = InducingPatches(Parameter(Z, name="inducing_loc"))
            else:
                Z = Parameter(Z, name="inducing_loc")

            if model_name == "sgpr":
                assert task == "regression"
                model = SGPR(self.train_data, kernel, Z)
            else:
                likelihood, P = self._setup_likelihood(task)

                if model_name == "msvgp":
                    model = SVGP(
                        kernel,
                        likelihood,
                        Z,
                        num_data=N,
                        whiten=False,
                        num_latent_gps=P,
                    )
                    K = Kuu(model.inducing_variable, kernel, jitter=default_jitter())
                    model.q_sqrt.assign(tf.linalg.cholesky(K)[tf.newaxis])
                elif model_name == "wsvgp":
                    model = SVGP(
                        kernel, likelihood, Z, num_data=N, whiten=True, num_latent_gps=P
                    )

                elif model_name == "lsvgp":
                    model = LSVGP(
                        kernel,
                        likelihood,
                        Z,
                        num_data=N,
                        preconditioner=preconditioner,
                        num_latent_gps=P,
                    )
                elif model_name in ["rsvgp", "rsvgp_n"]:
                    model = RSVGP(
                        kernel,
                        likelihood,
                        Z,
                        num_data=N,
                        preconditioner=preconditioner,
                        num_latent_gps=P,
                        num_probes=num_probes,
                    )
                    if model_name == "rsvgp_n":
                        set_trainable(model.q_T_sqrt, False)
                elif model_name == "lsvgp_full":
                    model = LSVGP(
                        kernel,
                        likelihood,
                        Z,
                        num_data=N,
                        q_diag=False,
                        preconditioner=preconditioner,
                        num_latent_gps=P,
                    )
                elif model_name in ["rsvgp_full", "rsvgp_full_n"]:
                    model = RSVGP(
                        kernel,
                        likelihood,
                        Z,
                        num_data=N,
                        q_diag=False,
                        preconditioner=preconditioner,
                        num_latent_gps=P,
                        num_probes=num_probes,
                    )
                    if model_name == "rsvgp_full_n":
                        set_trainable(model.q_T_sqrt, False)
                else:
                    raise ValueError(f"Unknown model class {model_name}")

            if isinstance(model, RSVGP) and debug_initialise_optimal_T:
                model.set_optimal_T()

        return model
