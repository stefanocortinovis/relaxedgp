from typing import Any, Dict, List, Optional, Tuple

import tensorflow as tf
from gpflow import default_float
from gpflow.base import TensorType
from gpflow.utilities import set_trainable
from gpflux.layers import GPLayer
from gpflux.sampling.sample import Sample

from ..typing import MinibatchModel


class GPLayerWrapper(GPLayer):
    """
    A wrapper around gpflux.layers.GPLayer that works with any SVGP
    parameterisation by taking a MinibatchModel as main argument.
    """

    def __init__(
        self,
        model: MinibatchModel,
        *,
        num_samples: Optional[int] = None,
        full_cov: bool = False,
        full_output_cov: bool = False,
        name: Optional[str] = None,
        verbose: bool = True,
    ):
        """
        :param model: The MinibatchModel to be wrapped.
        :param num_samples: The number of samples to draw when converting the
            :class:`~tfp.layers.DistributionLambda` into a `tf.Tensor`, see
            :meth:`_convert_to_tensor_fn`. Will be stored in the
            :attr:`num_samples` attribute.  If `None` (the default), draw a
            single sample without prefixing the sample shape (see
            :class:`tfp.distributions.Distribution`'s `sample()
            <https://www.tensorflow.org/probability/api_docs/python/tfp/distributions/Distribution#sample>`_
            method).
        :param full_cov: Sets default behaviour of calling this layer
            (:attr:`full_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to inputs.
            If `True`, predict full covariance over inputs.
        :param full_output_cov: Sets default behaviour of calling this layer
            (:attr:`full_output_cov` attribute):
            If `False` (the default), only predict marginals (diagonal
            of covariance) with respect to outputs.
            If `True`, predict full covariance over outputs.
        :param name: The name of this layer.
        :param verbose: The verbosity mode. Set this parameter to `True`
            to show debug information.
        """

        super().__init__(
            kernel=model.kernel,
            inducing_variable=model.inducing_variable,
            num_data=model.num_data,
            mean_function=model.mean_function,
            num_samples=num_samples,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            num_latent_gps=model.num_latent_gps,
            name=name,
            verbose=verbose,
        )

        # del self.q_mu
        # del self.q_sqrt
        set_trainable(self.q_mu, False)
        set_trainable(self.q_sqrt, False)

        self.model = model
        if self.model.likelihood is not None:
            set_trainable(self.model.likelihood, False)

    def predict(
        self,
        inputs: TensorType,
        *,
        full_cov: bool = False,
        full_output_cov: bool = False,
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        return self.model.predict_f(
            inputs, full_cov=full_cov, full_output_cov=full_output_cov
        )

    def call(
        self, inputs: TensorType, *args: List[Any], **kwargs: Dict[str, Any]
    ) -> tf.Tensor:
        outputs = super(GPLayer, self).call(inputs, *args, **kwargs)

        if kwargs.get("training"):
            log_prior = tf.add_n(
                [p.log_prior_density() for p in self.model.kernel.trainable_parameters]
            )
            loss = self.prior_kl() - log_prior
            loss_per_datapoint = loss / self.num_data

        else:
            # TF quirk: add_loss must always add a tensor to compile
            loss_per_datapoint = tf.constant(0.0, dtype=default_float())
        self.add_loss(loss_per_datapoint)

        # Metric names should be unique; otherwise they get overwritten if you
        # have multiple with the same name
        name = f"{self.name}_prior_kl" if self.name else "prior_kl"
        self.add_metric(loss_per_datapoint, name=name, aggregation="mean")

        return outputs

    def prior_kl(self) -> tf.Tensor:
        return self.model.prior_kl()

    def sample(self) -> Sample:
        """
        .. todo:: TODO: Need to extend gpflux.sample.efficient_sample() to
            different SVGP parameterisations.
        """
        raise NotImplementedError
