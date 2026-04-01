import logging
import traceback
from typing import Callable, Optional

import numpy as np
import tensorflow as tf
import wandb
from gpflow.base import RegressionData
from gpflow.utilities import set_trainable
from gpflux.models import DeepGP
from tensorflow.python.data.experimental.ops.prefetching_ops import prefetch_to_device
from tqdm import trange

from relaxedgp.natgrad import NaturalGradientInverseCholesky
from relaxedgp.schedules import NaturalGradientSchedule, ReduceLROnPlateau
from relaxedgp_experiments.dgp_utils.evaluate import training_loss
from relaxedgp_experiments.utils.log import log_metrics


def train(
    model: DeepGP,
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

    N, D = data[0].shape
    if batch_size is None:
        batch_size = N

    if len(tf.config.list_physical_devices("GPU")) > 0:
        prefetch = prefetch_to_device("/gpu:0")
    else:
        prefetch = lambda x: x

    opts = tf.data.Options()
    opts.deterministic = True

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

    first_gp_layer = model.f_layers[0].model
    if (
        hasattr(first_gp_layer, "q_T_sqrt")
        and not first_gp_layer.q_T_sqrt.trainable
        and natgrad_schedule is not None
    ):
        opt_q_T = NaturalGradientInverseCholesky(gamma=natgrad_schedule)

        @tf.function(jit_compile=True, reduce_retracing=True)
        def natgrad_step(train_batch):
            avg_num_steps = 0

            if opt_q_T.gamma.stopping_criterion is not None:
                avg_stopping_criterion = 0.0
            else:
                avg_stopping_criterion = None

            for _, f_layer in enumerate(model.f_layers):
                gp_layer = f_layer.model
                K_tilde = gp_layer.K_tilde()
                num_steps, stopping_criterion = opt_q_T.minimize(
                    K_tilde,
                    gp_layer.q_T_sqrt,
                    gp_layer,
                    train_batch,
                )

                avg_num_steps += num_steps
                if avg_stopping_criterion is not None:
                    avg_stopping_criterion += stopping_criterion

            avg_num_steps /= len(model.f_layers)
            if avg_stopping_criterion is not None:
                avg_stopping_criterion /= len(model.f_layers)

            return avg_num_steps, avg_stopping_criterion
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
    else:
        scheduler_Z = None

    @tf.function(jit_compile=True, reduce_retracing=True)
    def adam_step(train_batch):
        with tf.GradientTape() as tape:
            loss = -model.elbo(train_batch)
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
            for i, (grad, _) in enumerate(grads_Z):
                metrics[f"opt/grad_norm_Z_{i}"] = tf.norm(grad).numpy().item()

        if callback is not None:
            metrics.update(callback())

        return metrics

    for f_layer in model.f_layers:
        gp_layer = f_layer.model
        set_trainable(gp_layer.inducing_variable, False)

    lmls = []
    logger.info(f"Training DGP model for {n_iter} iterations.")
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
                for f_layer in model.f_layers:
                    gp_layer = f_layer.model
                    set_trainable(gp_layer.inducing_variable, True)
                adam_step = tf.function(
                    adam_step.python_function,
                    jit_compile=True,
                    reduce_retracing=True,
                )

            if debug_movement_Z:
                Z_current = [
                    f_layer.model.inducing_variable.inducing_variable.Z.numpy().copy()
                    for f_layer in model.f_layers
                ]

            metrics = train_step(train_batch)
            loss = metrics["train/loss"]

            if debug_movement_Z:
                for i, f_layer in enumerate(model.f_layers):
                    gp_layer = f_layer.model
                    Z_new = gp_layer.inducing_variable.inducing_variable.Z.numpy()
                    movement = np.linalg.norm(Z_new - Z_current[i])
                    metrics[f"opt/movement_Z_{i}"] = movement

            if debug_full_loss:
                lmls.append(
                    [
                        step + 1,
                        loss,
                        training_loss(model, data, batch_size=batch_size)
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
                scheduler_Z.update(opt_Z, loss)

            if log_wandb:
                metrics["opt/lr_adam"] = opt_adam.learning_rate.numpy().item()
                metrics["opt/lr_adam_Z"] = opt_Z.learning_rate.numpy().item()

                if opt_q_T is not None:
                    # TODO: this is not logging the right thing
                    metrics["opt/lr_natgrad"] = opt_q_T.gamma(step)

                log_metrics(metrics, step=step + 1, logger=logger, log_wandb=log_wandb)
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
