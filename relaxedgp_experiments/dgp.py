import logging
import os
import time
from typing import Any, Dict, List, Optional

import hydra
import tensorflow as tf
import wandb
from gpflow.base import InputData, OutputData, RegressionData
from gpflow.config import default_jitter
from gpflow.covariances import Kuu
from gpflow.models import SVGP
from gpflow.utilities import set_trainable
from gpflux.models import DeepGP
from omegaconf import DictConfig

from relaxedgp.deep.architectures import build_constant_input_dim_deep_gp
from relaxedgp.models import LSVGP, RSVGP
from relaxedgp.schedules import NaturalGradientSchedule
from relaxedgp_experiments.base import Experiment
from relaxedgp_experiments.dgp_utils.evaluate import evaluation_metrics, training_loss
from relaxedgp_experiments.dgp_utils.log import get_callback
from relaxedgp_experiments.dgp_utils.plot import _plot_model
from relaxedgp_experiments.dgp_utils.train import train
from relaxedgp_experiments.utils.data import Dataset


# TODO: generalise to Z_init other than kmeans
class DGPExperiment(Experiment):
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
        num_layers: int = 1,
        inner_layer_qsqrt_factor: float = 1.0,
        likelihood_noise_variance: float = 1.0,
        hidden_kernel_variance: float = 1e-6,
        preconditioner: str = "identity",
        logger: Optional[logging.Logger] = None,
        debug_initialise_optimal_T: bool = False,
        plot_kwargs: Optional[Dict[str, Any]] = None,
        eval_num_samples: int = 1,
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
            num_layers,
            task,
            kernel_name,
            init_Z_method=None,  # TODO
            preconditioner=preconditioner,
            debug_initialise_optimal_T=debug_initialise_optimal_T,
            inner_layer_qsqrt_factor=inner_layer_qsqrt_factor,
            likelihood_noise_variance=likelihood_noise_variance,
            hidden_kernel_variance=hidden_kernel_variance,
            num_probes=num_probes,
        )
        self.plot_kwargs = plot_kwargs or dict()
        self.eval_num_samples = eval_num_samples

    def run(self, **kwargs: Any) -> List[float]:
        self.logger.info("Running experiment...")
        lmls = self.train_model(**kwargs)
        self.save_model()
        if (self.D == 1 and self.task == "regression") or (
            self.D == 2 and self.task == "classification"
        ):
            self.plot_model()
        self.logger.info("Experiment finished.")
        return lmls

    def evaluate_model(self) -> Dict:
        if self.test_data is None:
            return dict()
        N_test = self.test_data[0].shape[0]
        if self.eval_num_samples > N_test:
            batch_size = None
        else:
            batch_size = N_test // self.eval_num_samples
        return evaluation_metrics(
            self.model,
            self.test_data,
            batch_size=batch_size,
            task=self.task,
            num_samples=self.eval_num_samples,
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
        reduce_on_plateau: bool = False,
        reduce_on_plateau_Z: bool = False,
        debug_full_loss: bool = False,
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
            n_iter=n_iter,
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

            for i, f_layer in enumerate(self.model.f_layers):
                gp_layer = f_layer.model
                wandb.log(
                    {
                        f"model/kernel_variance_{i}": gp_layer.kernel.kernel.variance.numpy().item()  # noqa: E501
                    }
                )
            run.finish()

        return lmls

    @staticmethod
    def setup_from_config(
        cfg: DictConfig, log_dir: str, logger: Optional[logging.Logger] = None
    ) -> "Experiment":
        num_probes = cfg.model.get("num_probes", None)
        if num_probes is not None and num_probes > cfg.model.M:
            num_probes = None

        return DGPExperiment(
            log_dir=log_dir,
            dataset_name=cfg.dataset.name,
            model_name=cfg.model.name,
            task=cfg.dataset.task,
            kernel_name=cfg.model.kernel_name,
            train_only=cfg.dataset.train_only,
            origin=cfg.dataset.origin,
            normalize_data=cfg.dataset.normalize,
            M=cfg.model.M,
            num_layers=cfg.model.num_layers,
            inner_layer_qsqrt_factor=cfg.model.inner_layer_qsqrt_factor,
            likelihood_noise_variance=cfg.model.likelihood_noise_variance,
            hidden_kernel_variance=cfg.model.hidden_kernel_variance,
            preconditioner=cfg.model.preconditioner,
            logger=logger,
            debug_initialise_optimal_T=cfg.debug.initialise_optimal_T,
            eval_num_samples=cfg.eval.num_samples,
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
            reduce_on_plateau=cfg.training.reduce_on_plateau,
            reduce_on_plateau_Z=cfg.training.reduce_on_plateau_Z,
            debug_full_loss=cfg.debug.full_loss,
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

    def _setup_model(
        self,
        model_name: str,
        M: Optional[int],
        num_layers: int,
        task: str = "regression",
        kernel_name: str = "squared_exponential",
        init_Z_method: str = "random",
        preconditioner: str = "identity",
        debug_initialise_optimal_T: bool = False,
        inner_layer_qsqrt_factor: float = 1.0,
        likelihood_noise_variance: float = 1.0,
        hidden_kernel_variance: float = 1e-6,
        num_probes: Optional[int] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> DeepGP:
        if model_name == "msvgp":
            base_model_class = SVGP
            model_kwargs = {"whiten": False}
        elif model_name == "wsvgp":
            base_model_class = SVGP
            model_kwargs = {"whiten": True}
        elif model_name == "lsvgp":
            base_model_class = LSVGP
            model_kwargs = {
                "q_diag": True,
                "preconditioner": preconditioner,
            }
        elif model_name in ["rsvgp", "rsvgp_n"]:
            base_model_class = RSVGP
            model_kwargs = {
                "q_diag": True,
                "preconditioner": preconditioner,
                "num_probes": num_probes,
            }
        elif model_name == "lsvgp_full":
            base_model_class = LSVGP
            model_kwargs = {
                "q_diag": False,
                "preconditioner": preconditioner,
            }
        elif model_name in ["rsvgp_full", "rsvgp_full_n"]:
            base_model_class = RSVGP
            model_kwargs = {
                "q_diag": False,
                "preconditioner": preconditioner,
                "num_probes": num_probes,
            }
        else:
            raise ValueError(f"Unknown model class {model_name}")

        if model_kwargs is None:
            model_kwargs = dict()

        N = self.N
        if M is None:
            M = N

        model = build_constant_input_dim_deep_gp(
            base_model_class,
            self.X_train,
            num_layers=num_layers,
            num_inducing=M,
            inner_layer_qsqrt_factor=inner_layer_qsqrt_factor,
            likelihood_noise_variance=likelihood_noise_variance,
            hidden_kernel_variance=hidden_kernel_variance,
            task=self.task,
            kernel_name=kernel_name,
            **model_kwargs,
        )

        if model_name == "msvgp":
            for f_layer in model.f_layers:
                gp_layer = f_layer.model
                K = Kuu(
                    gp_layer.inducing_variable, gp_layer.kernel, jitter=default_jitter()
                )
                # TODO: right shape with shared?
                gp_layer.q_sqrt.assign(tf.linalg.cholesky(K)[tf.newaxis])
        elif model_name in ["rsvgp_n", "rsvgp_full_n"]:
            for f_layer in model.f_layers:
                gp_layer = f_layer.model
                set_trainable(gp_layer.q_T_sqrt, False)

        if (
            model_name in ["rsvgp", "rsvgp_n", "rsvgp_full", "rsvgp_full_n"]
            and debug_initialise_optimal_T
        ):
            for f_layer in model.f_layers:
                gp_layer = f_layer.model
                gp_layer.set_optimal_T()

        return model
