import logging
import traceback
from typing import Callable, Optional

import gpflow
import numpy as np
import tensorflow as tf
import wandb
from gpflow.base import RegressionData
from gpflow.kernels import Convolutional
from gpflow.utilities import set_trainable
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device
from tqdm import trange

from relaxedgp.models import RSVGP
from relaxedgp.natgrad import NaturalGradientInverseCholesky
from relaxedgp.schedules import NaturalGradientSchedule, ReduceLROnPlateau
from relaxedgp.typing import SGPR, FullBatchModel, MinibatchModel, Model
from relaxedgp_experiments.utils.evaluate import training_loss
from relaxedgp_experiments.utils.log import log_metrics


def train(
    model: Model,
    data: RegressionData,
    n_iter: int = 10000,
    adam_lr: float = 1e-3,
    natgrad_schedule: Optional[NaturalGradientSchedule] = None,
    train_Z: bool = False,
    batch_size: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
    log_wandb: bool = False,
    callback: Optional[Callable] = None,
    reduce_on_plateau: bool = False,
    reduce_on_plateau_Z: bool = False,
    debug_force_bfgs: bool = False,
    debug_full_loss: bool = False,
    warmup_Z: int = 0,
    adam_lr_Z: Optional[float] = None,
    adam_beta_1_Z: float = 0.9,
    debug_grad_Z: bool = False,
    debug_movement_Z: bool = False,
    verbose: bool = True,
) -> Optional[np.ndarray]:
    if logger is None:
        logger = logging.getLogger(__name__)

    if debug_force_bfgs:
        assert batch_size is None

    if isinstance(model, FullBatchModel) or debug_force_bfgs:
        logger.info(f"Training {model.kernel.name}.")

        if not train_Z and isinstance(model, (SGPR, MinibatchModel)):
            set_trainable(model.inducing_variable, False)

        if isinstance(model, MinibatchModel):  # debug_force_bfgs = True
            loss = lambda: model.training_loss(data)
        else:
            loss = model.training_loss

        opt = gpflow.optimizers.Scipy()
        opt_result = opt.minimize(
            loss,
            model.trainable_variables,
            callback=callback,
        )
        callback(opt_result)

    else:
        N, D = data[0].shape
        if batch_size is None:
            batch_size = N

        if len(tf.config.list_physical_devices("GPU")) > 0:
            prefetch = prefetch_to_device("/gpu:0")
        else:
            prefetch = lambda x: x

        opts = tf.data.Options()
        opts.deterministic = False

        trainloader = iter(
            prefetch(
                tf.data.Dataset.from_tensor_slices(data)
                .shuffle(N, reshuffle_each_iteration=True)
                .repeat()
                .batch(batch_size, drop_remainder=True)
                # .cache()
                .prefetch(tf.data.AUTOTUNE)
            ).with_options(opts)
        )

        if adam_lr_Z is None:
            adam_lr_Z = adam_lr

        opt_adam = tf.keras.optimizers.Adam(adam_lr)
        opt_Z = tf.keras.optimizers.Adam(adam_lr_Z, beta_1=adam_beta_1_Z)

        if (
            hasattr(model, "q_T_sqrt")
            and not model.q_T_sqrt.trainable
            and natgrad_schedule is not None
        ):
            opt_q_T = NaturalGradientInverseCholesky(gamma=natgrad_schedule)

            @tf.function(jit_compile=True, reduce_retracing=True)
            def natgrad_step(train_batch):
                K_tilde = model.K_tilde()
                return opt_q_T.minimize(
                    K_tilde,
                    model.q_T_sqrt,
                    model,
                    train_batch,
                )
        else:
            opt_q_T = None

            def natgrad_step(train_batch):
                return None, None

        if reduce_on_plateau:
            # TODO: allow to set
            scheduler = ReduceLROnPlateau(
                factor=0.95,
                patience=1000,
                min_delta=0.0,
                cooldown=0,
                min_lr=1e-6,
            )
        elif isinstance(model.kernel, Convolutional):
            boundaries = [30000 * i for i in range(1, 51)]
            values = [adam_lr * 10 ** -(i / 3) for i in range(len(boundaries))]
            scheduler = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries[:-1], values
            )
        else:
            scheduler = None

        if reduce_on_plateau_Z:
            scheduler_Z = ReduceLROnPlateau(
                factor=0.95,
                patience=100,
                min_delta=0.0,
                cooldown=0,
                min_lr=1e-6,
            )
        elif reduce_on_plateau:
            scheduler_Z = ReduceLROnPlateau(
                factor=0.95,
                patience=1000,
                min_delta=0.0,
                cooldown=0,
                min_lr=1e-6,
            )  # same as scheduler, used in DGPs with no reduce_on_plateau_Z
        elif isinstance(model.kernel, Convolutional):
            scheduler_Z = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
                boundaries[:-1], values
            )
        else:
            scheduler_Z = None

        @tf.function(jit_compile=True, reduce_retracing=True)
        def adam_step(train_batch):
            with tf.GradientTape() as tape:
                loss = model.training_loss(train_batch)
            grads = tape.gradient(loss, model.trainable_variables)

            grads_vars = []
            grads_Z = []
            for grad, var in zip(grads, model.trainable_variables):
                if "inducing_loc" in var.name:
                    grads_Z.append((grad, var))
                else:
                    grads_vars.append((grad, var))

            opt_adam.apply_gradients(grads_vars)

            if grads_Z:
                opt_Z.apply_gradients(grads_Z)
            return loss, grads_Z

        def train_step(train_batch):
            metrics = dict()

            loss, grads_Z = adam_step(train_batch)
            metrics["train/loss"] = loss.numpy().item()
            num_updates, criterion = natgrad_step(train_batch)
            metrics["opt/number_natgrad_updates"] = num_updates
            metrics["opt/natgrad_criterion"] = (
                criterion.numpy().item() if criterion is not None else None
            )

            if debug_grad_Z:
                for grad, _ in grads_Z:
                    metrics["opt/grad_norm_Z"] = tf.norm(grad).numpy().item()

            if callback is not None:
                metrics.update(callback())

            return metrics

        set_trainable(model.inducing_variable, False)

        lmls = []
        logger.info(f"Training {model.kernel.name} for {n_iter} iterations.")
        try:
            step = 0
            metrics = dict()
            num_updates, criterion = natgrad_step(next(trainloader))
            metrics["opt/number_natgrad_updates"] = num_updates
            metrics["opt/natgrad_criterion"] = (
                criterion.numpy().item() if criterion is not None else None
            )

            if log_wandb:
                if callback is not None:
                    metrics.update(callback())

                wandb.log(metrics, step=0)

            for step, train_batch in zip(
                trange(n_iter, position=0, leave=True, disable=not verbose), trainloader
            ):
                if step == warmup_Z and train_Z:
                    set_trainable(model.inducing_variable, True)
                    adam_step = tf.function(
                        adam_step.python_function,
                        jit_compile=True,
                        reduce_retracing=True,
                    )

                if debug_movement_Z:
                    Z_current = model.inducing_variable.Z.numpy().copy()

                metrics = train_step(train_batch)
                loss = metrics["train/loss"]

                if isinstance(model, RSVGP):
                    metrics["model/s_tilde"] = np.mean(model.q_sqrt.numpy())

                if debug_movement_Z:
                    metrics["opt/Z_movement"] = np.linalg.norm(
                        model.inducing_variable.Z.numpy() - Z_current
                    )

                if debug_full_loss:
                    lmls.append(
                        [
                            step + 1,
                            loss,
                            training_loss(
                                model, data, batch_size=None
                            )  # NOTE: this should only be used for small datasets!
                            .numpy()
                            .item(),
                        ]
                    )
                else:
                    lmls.append([step + 1, loss])

                if np.isnan(loss):
                    logger.info(
                        f"Training failed at iteration {step + 1} with error: "
                        "loss is NaN."
                    )
                    break

                if scheduler is not None:
                    if reduce_on_plateau:
                        scheduler.update(opt_adam, loss)
                    else:
                        opt_adam.learning_rate = scheduler(step)
                if (scheduler_Z is not None) and (step >= warmup_Z):
                    if reduce_on_plateau_Z or reduce_on_plateau:
                        scheduler_Z.update(opt_Z, loss)
                    else:
                        opt_Z.learning_rate = scheduler_Z(step)

                if log_wandb:
                    metrics["opt/lr_adam"] = opt_adam.learning_rate.numpy().item()
                    metrics["opt/lr_adam_Z"] = opt_Z.learning_rate.numpy().item()

                    if opt_q_T is not None:
                        # TODO: this is not logging the right thing
                        metrics["opt/lr_natgrad"] = opt_q_T.gamma(step)

                    log_metrics(
                        metrics, step=step + 1, logger=logger, log_wandb=log_wandb
                    )
                else:
                    metrics.pop("opt/number_natgrad_updates", None)
                    metrics.pop("opt/natgrad_criterion", None)

                    if "test/nlpd" in metrics:
                        log_metrics(
                            metrics, step=step + 1, logger=logger, log_wandb=log_wandb
                        )

        except KeyboardInterrupt:
            logger.info(f"Training interrupted at iteration {step + 1}.")
        except Exception:
            logger.info(
                f"Training failed at iteration {step + 1} with error: "
                f"{traceback.print_exc()}"
            )
        else:
            logger.info("Training completed.")

        return np.array(lmls)
