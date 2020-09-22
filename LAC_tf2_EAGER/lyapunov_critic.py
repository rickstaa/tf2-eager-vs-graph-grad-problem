"""Contains the tensorflow critic.
"""

import tensorflow as tf


class LyapunovCritic(tf.keras.Model):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        name,
        log_std_min=-20,
        log_std_max=2.0,
        trainable=True,
        seed=None,
        **kwargs,
    ):
        """Lyapunov Critic network.

        Args:
            obs_dim (int): The dimension of the observation space.

            act_dim (int): The dimension of the action space.

            hidden_sizes (list): Array containing the sizes of the hidden layers.

            name (str): The keras module name.

            log_std_min (int, optional): The min log_std. Defaults to -20.

            log_std_max (float, optional): The max log_std. Defaults to 2.0.

            trainable (bool, optional): Whether the weights of the network layers should
                be trainable. Defaults to True.

            seed (int, optional): The random seed. Defaults to None.
        """
        super().__init__(name=name, **kwargs)

        # Get class parameters
        self.s_dim = obs_dim
        self.a_dim = act_dim
        self._seed = seed
        self._initializer = tf.keras.initializers.GlorotUniform(
            seed=self._seed
        )  # Seed weights initializer

        # Create network layers
        self.net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(
                    dtype=tf.float32, input_shape=(obs_dim + act_dim), name="input",
                )
            ]
        )
        for i, hidden_size_i in enumerate(hidden_sizes):
            self.net.add(
                tf.keras.layers.Dense(
                    hidden_size_i,
                    activation="relu",
                    name=name + "/{}".format(i),
                    trainable=trainable,
                    kernel_initializer=self._initializer,
                )
            )

    @tf.function
    def call(self, inputs):
        """Perform forward pass."""

        # Perform forward pass through input layers
        net_out = self.net(tf.concat(inputs, axis=-1))

        # Return result
        return tf.expand_dims(
            tf.reduce_sum(tf.math.square(net_out), axis=1), axis=1
        )  # L(s,a)
