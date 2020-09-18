"""Contains the tensorflow actor.
"""

import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector


class SquashedGaussianActor(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        name,
        log_std_min=-20,
        log_std_max=2.0,
        seed=None,
        **kwargs
    ):
        # TODO: Check if name results in problem
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max
        self.s_dim = obs_dim
        self.a_dim = act_dim
        self.tfp_seed = tfp.util.SeedStream(seed, salt="random_beta")

        # Create fully connected layers
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=(self.s_dim), name="input"
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.net.add(
                tf.keras.layers.Dense(
                    hidden_size_i, activation="relu", name="l{}".format(i + 1)
                )
            )

        # Create Mu and log sigma output layers
        self.mu = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(act_dim, activation=None, name="mu"),
            ]
        )
        self.log_sigma = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=hidden_sizes[-1]
                ),
                tf.keras.layers.Dense(act_dim, activation=None, name="log_sigma"),
            ]
        )

    @tf.function
    def call(self, inputs):
        """Perform forward pass."""

        # Retrieve inputs
        obs = inputs

        # Perform forward pass through fully connected layers
        net_out = self.net(obs)

        # Calculate mu and log_sigma
        mu = self.mu(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = tf.clip_by_value(log_sigma, self._log_std_min, self._log_std_max)

        # Perform re-parameterization trick
        sigma = tf.exp(log_sigma)

        # Create bijectors (Used in the re-parameterization trick)
        squash_bijector = SquashBijector()
        affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(sigma))

        # Sample from the normal distribution and calculate the action
        batch_size = tf.shape(input=obs)[0]
        base_distribution = tfp.distributions.MultivariateNormalDiag(
            loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
        )
        epsilon = base_distribution.sample(batch_size, seed=self.tfp_seed())
        raw_action = affine_bijector.forward(epsilon)
        clipped_a = squash_bijector.forward(raw_action)

        # Transform distribution back to the original policy distribution
        reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
        distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution, bijector=reparm_trick_bijector
        )
        clipped_mu = squash_bijector.forward(mu)
        return clipped_a, clipped_mu, distribution.log_prob(clipped_a)
