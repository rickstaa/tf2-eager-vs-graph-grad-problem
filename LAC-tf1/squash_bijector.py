"""Creates a squash (tanh) bijector for using the parameterization trick.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class SquashBijector(tfp.bijectors.Bijector):
    """A squash bijector used to keeps track of the distribution properties when the
    distribution is transformed using the tanh squash function."""

    def __init__(self, validate_args=False, name="tanh"):
        super(SquashBijector, self).__init__(
            forward_min_event_ndims=0, validate_args=validate_args, name=name
        )

    def _forward(self, x):
        return tf.nn.tanh(x)
        # return x

    def _inverse(self, y):
        return tf.atanh(y)

    def _forward_log_det_jacobian(self, x):
        return 2.0 * (np.log(2.0) - x - tf.nn.softplus(-2.0 * x))
