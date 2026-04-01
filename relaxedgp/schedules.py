from abc import abstractmethod
from typing import Optional

import numpy as np
import tensorflow as tf
from gpflow.config import default_float

from .stopping_criteria import PredictiveVarianceCriterion, StoppingCriterion


class ReduceLROnPlateau:
    """Reduce learning rate when a metric has stopped improving.

    Args:
        factor: Float. Factor by which the learning rate will be reduced.
            `new_lr = lr * factor`.
        patience: Integer. Number of epochs with no improvement after which
            learning rate will be reduced.
        min_delta: Float. Threshold for measuring the new optimum, to only focus
            on significant changes.
        cooldown: Integer. Number of epochs to wait before resuming normal
            operation after the learning rate has been reduced.
        min_lr: Float. Lower bound on the learning rate.
    """

    def __init__(
        self,
        factor: float = 0.1,
        patience: int = 10,
        min_delta: float = 1e-4,
        cooldown: int = 0,
        min_lr: float = 0.0,
    ) -> None:
        if factor >= 1.0:
            raise ValueError(
                "ReduceLROnPlateau does not support a factor >= 1.0. "
                f"Received factor={factor}"
            )

        self.factor = factor
        self.min_lr = min_lr
        self.min_delta = min_delta
        self.patience = patience
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self._reset()

    def _reset(self) -> None:
        """Resets wait counter and cooldown counter."""
        self.best = np.Inf
        self.cooldown_counter = 0
        self.wait = 0

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_counter > 0

    def update(
        self,
        optim: tf._optimizers.Optimizer,
        value: float,
    ):
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.wait = 0

        if np.less(value, self.best - self.min_delta):
            self.best = value
            self.wait = 0
        elif not self.in_cooldown:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = optim.learning_rate.numpy().item()

                if old_lr > np.float32(self.min_lr):
                    new_lr = old_lr * self.factor
                    new_lr = max(new_lr, self.min_lr)
                    optim.learning_rate = new_lr
                    self.cooldown_counter = self.cooldown
                    self.wait = 0


class NaturalGradientSchedule:
    def __init__(
        self,
        max_steps: int,
        stopping_criterion: Optional[StoppingCriterion] = None,
    ) -> None:
        self.max_steps = max_steps
        self.stopping_criterion = stopping_criterion

    @abstractmethod
    def __call__(self, step: int) -> float:
        raise NotImplementedError


class ConstantSchedule(NaturalGradientSchedule):
    def __init__(
        self,
        max_steps: int,
        value: float,
        stopping_criterion: Optional[StoppingCriterion] = None,
        line_search: bool = False,
    ) -> None:
        super().__init__(max_steps, stopping_criterion)
        self.value = value
        if line_search and not isinstance(
            stopping_criterion, PredictiveVarianceCriterion
        ):
            raise ValueError(
                "Line search is only supported with predictive variance criterion."
            )
        self.line_search = line_search

    def __call__(self, step: int) -> float:
        return tf.constant(self.value, dtype=default_float())


class ExponentialSchedule(NaturalGradientSchedule):
    def __init__(
        self,
        max_steps: int,
        init_value: float,
        transition_steps: int,
        decay_rate: float,
        staircase: bool = False,
        end_value: float = 0.0,
        stopping_criterion: Optional[StoppingCriterion] = None,
    ) -> None:
        super().__init__(max_steps, stopping_criterion)
        self.init_value = tf.convert_to_tensor(init_value, dtype=default_float())
        self.transition_steps = transition_steps
        self.decay_rate = tf.convert_to_tensor(decay_rate, default_float())
        self.staircase = staircase
        self.end_value = tf.convert_to_tensor(end_value, dtype=default_float())

    def __call__(self, step: int) -> tf.Tensor:
        rate = step / self.transition_steps
        if self.staircase:
            rate = tf.floor(rate)
        decayed_value = self.init_value * (self.decay_rate**rate)
        if tf.sign(1.0 - self.decay_rate) * (decayed_value - self.end_value) < 0.0:
            return self.end_value
        return decayed_value


class LogLinearSchedule(NaturalGradientSchedule):
    def __init__(
        self,
        max_steps: int,
        init_value: float,
        transition_steps: int,
        end_value: float = 1.0,
        stopping_criterion: Optional[StoppingCriterion] = None,
    ) -> None:
        super().__init__(max_steps, stopping_criterion)
        self.log_init_value = tf.convert_to_tensor(
            np.log(init_value), dtype=default_float()
        )
        self.end_value = tf.convert_to_tensor(end_value, dtype=default_float())
        self.linear_slope = (tf.math.log(self.end_value) - self.log_init_value) / (
            transition_steps - 1
        )
        self.transition_steps = transition_steps

    def __call__(self, step: int) -> tf.Tensor:
        if step < self.transition_steps:
            return tf.cast(
                tf.exp(
                    self.log_init_value
                    + (self.linear_slope * tf.cast(step, default_float()))
                ),
                dtype=default_float(),
            )
        return self.end_value
