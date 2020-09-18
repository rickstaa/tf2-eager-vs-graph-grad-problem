"""Eager mode disabled grad debug script
Small debug script to validate whether the computed Actor and Critic gradients in the
new eager mode are equal to the computed gradients when using disable_eager_execution.

The Agent is based on this paper: http://arxiv.org/abs/2004.14288
"""

import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

# Disable GPU if requested
# NOTE: Done so i can run both scripts in a debugger side by side
tf.config.set_visible_devices([], "GPU")

# Disable eager
tf.compat.v1.disable_eager_execution()

####################################################
# Script parameters ################################
####################################################
S_DIM = 2  # Observation space dimension
A_DIM = 2  # Observation space dimension
BATCH_SIZE = 256  # Replay buffer batch size
LOG_SIGMA_MIN_MAX = (-20, 2)  # Range of log std coming out of the GA network
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier
ALPHA_3 = 0.2  # The value of the stability condition multiplier
GAMMA = 0.9  # Discount factor
ALPHA = 0.99  # The initial value for the entropy lagrance multiplier
LAMBDA = 0.99  # Initial value for the lyapunov constraint lagrance multiplier
NETWORK_STRUCTURE = {
    "critic": [6, 6],
    "actor": [6, 6],
}  # The network structure of the agent.
POLYAK = 0.995  # Decay rate used in the polyak averaging
LR_A = 1e-4  # The actor learning rate
LR_L = 3e-4  # The lyapunov critic
LR_LAG = 1e-4  # The lagrance multiplier learning rate


# Gradient settings
GRAD_SCALE_FACTOR = 500  # Scale the grads by a factor to make differences more visible

####################################################
# Seed random number generators ####################
####################################################
RANDOM_SEED = 0  # The random seed

# Set random seed to get comparable results for each run
# NOTE: https://stackoverflow.com/questions/32419510/how-to-get-reproducible-results-in-keras
if RANDOM_SEED is not None:

    # Set random seeds
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    TFP_SEED_STREAM = tfp.util.SeedStream(RANDOM_SEED, salt="tfp_1")


####################################################
# Used helper functions ############################
####################################################
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
        return 2.0 * (tf.math.log(2.0) - x - tf.nn.softplus(-2.0 * x))


def retrieve_weights_biases():
    """Returns the current weight and biases from the (target) Gaussian Actor and
    Lyapunov critic.

    Returns:
        tuple: Tuple containing the weight and biases dictionaries of the (target)
        Gaussian Actor andLyapunov critic
    """

    # Retrieve initial network weights
    ga_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/gaussian_actor",
    )
    ga_target_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Actor/gaussian_actor",
    )
    lc_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/lyapunov_critic",
    )
    lc_target_vars = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/Actor/lyapunov_critic",
    )
    ga_weights_biases = policy.sess.run(ga_vars)
    ga_target_weights_biases = policy.sess.run(ga_target_vars)
    lc_weights_biases = policy.sess.run(lc_vars)
    lc_target_weights_biases = policy.sess.run(lc_target_vars)
    ga_weights_biases = {
        "l1/weights": ga_weights_biases[0],
        "l1/bias": ga_weights_biases[1],
        "l2/weights": ga_weights_biases[2],
        "l2/bias": ga_weights_biases[3],
        "mu/weights": ga_weights_biases[4],
        "mu/bias": ga_weights_biases[5],
        "log_sigma/weights": ga_weights_biases[6],
        "log_sigma/bias": ga_weights_biases[7],
    }
    ga_target_weights_biases = {
        "l1/weights": ga_target_weights_biases[0],
        "l1/bias": ga_target_weights_biases[1],
        "l2/weights": ga_target_weights_biases[2],
        "l2/bias": ga_target_weights_biases[3],
        "mu/weights": ga_target_weights_biases[4],
        "mu/bias": ga_target_weights_biases[5],
        "log_sigma/weights": ga_target_weights_biases[6],
        "log_sigma/bias": ga_target_weights_biases[7],
    }
    lc_weights_biases = {
        "l1/w1_s": lc_weights_biases[0],
        "l1/w1_a": lc_weights_biases[1],
        "l1/b1": lc_weights_biases[2],
        "l2/weights": lc_weights_biases[3],
        "l2/bias": lc_weights_biases[4],
    }
    lc_target_weights_biases = {
        "l1/w1_s": lc_target_weights_biases[0],
        "l1/w1_a": lc_target_weights_biases[1],
        "l1/b1": lc_target_weights_biases[2],
        "l2/weights": lc_target_weights_biases[3],
        "l2/bias": lc_target_weights_biases[4],
    }

    # Return weights and biases
    return (
        ga_weights_biases,
        ga_target_weights_biases,
        lc_weights_biases,
        lc_target_weights_biases,
    )


####################################################
# Agent class ######################################
####################################################
class LAC(object):
    """The lyapunov actor critic agent.
    """

    def __init__(self):

        # Save action and observation space as members
        self.a_dim = A_DIM
        self.s_dim = S_DIM

        # Set algorithm parameters as class objects
        self.network_structure = NETWORK_STRUCTURE
        self.polyak = POLYAK

        # Create network seeds
        self.ga_seeds = [
            RANDOM_SEED,
            TFP_SEED_STREAM(),
        ]  # [weight init seed, sample seed]
        self.ga_target_seeds = [
            RANDOM_SEED + 1,
            TFP_SEED_STREAM(),
        ]  # [weight init seed, sample seed]
        self.lya_ga_target_seeds = [
            RANDOM_SEED,
            TFP_SEED_STREAM(),
        ]  # [weight init seed, sample seed]
        self.lc_seed = RANDOM_SEED + 2  # Weight init seed
        self.lc_target_seed = RANDOM_SEED + 3  # Weight init seed

        # Determine target entropy
        self.target_entropy = -A_DIM  # lower bound of the policy entropy

        # Create tensorflow session
        self.sess = tf.compat.v1.Session()

        # Create networks, optimizers and variables inside the Actor scope
        with tf.compat.v1.variable_scope("Actor"):

            # Create observations placeholders
            self.S = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], "s")
            self.S_ = tf.compat.v1.placeholder(tf.float32, [None, self.s_dim], "s_")
            self.a_input = tf.compat.v1.placeholder(
                tf.float32, [None, self.a_dim], "a_input"
            )
            self.a_input_ = tf.compat.v1.placeholder(
                tf.float32, [None, self.a_dim], "a_input_"
            )
            self.R = tf.compat.v1.placeholder(tf.float32, [None, 1], "r")
            self.terminal = tf.compat.v1.placeholder(tf.float32, [None, 1], "terminal")

            # Create Learning rate placeholders
            self.LR_A = tf.compat.v1.placeholder(tf.float32, None, "LR_A")
            self.LR_lag = tf.compat.v1.placeholder(tf.float32, None, "LR_lag")
            self.LR_L = tf.compat.v1.placeholder(tf.float32, None, "LR_L")

            # Create lagrance multiplier placeholders
            self.log_labda = tf.compat.v1.get_variable(
                "lambda", None, tf.float32, initializer=tf.math.log(LAMBDA)
            )
            self.log_alpha = tf.compat.v1.get_variable(
                "alpha", None, tf.float32, initializer=tf.math.log(ALPHA)
            )
            self.labda = tf.clip_by_value(tf.exp(self.log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(self.log_alpha)

            ###########################################
            # Create Networks #########################
            ###########################################

            # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
            self.a, self.deterministic_a, self.a_dist, self.epsilon = self._build_a(
                self.S, seeds=self.ga_seeds
            )
            self.l = self._build_l(self.S, self.a_input, seed=self.lc_seed)
            self.log_pis = log_pis = self.a_dist.log_prob(
                self.a
            )  # Gaussian actor action log_probability

            # Retrieve GA and LC network parameters
            self.a_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/gaussian_actor"
            )
            self.l_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope="Actor/lyapunov_critic",
            )

            # Create EMA target network update policy (Soft replacement)
            ema = tf.train.ExponentialMovingAverage(decay=(self.polyak))

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [
                ema.apply(self.a_params),
                ema.apply(self.l_params),
            ]

            # Create GA and LC target networks
            # Don't get optimized but get updated according to the EMA of the main
            # networks
            self.a_, self.deterministic_a_, self.a_dist_, self.epsilon_ = self._build_a(
                self.S_,
                reuse=True,
                custom_getter=ema_getter,
                seeds=self.ga_target_seeds,
            )
            self.log_pis_ = self.a_dist_.log_prob(
                self.a_
            )  # Gaussian actor action log_probability
            self.l_ = self._build_l(
                self.S_,
                self.a_,
                reuse=True,
                custom_getter=ema_getter,
                seed=self.lc_target_seed,
            )

            # Create Networks for the (fixed) lyapunov temperature boundary
            # DEBUG: This graph has the same parameters as the original gaussian actor
            # but now it receives the next state. This was needed as the target network
            # uses exponential moving average.
            (
                self.lya_a_,
                self.lya_deterministic_a_,
                self.lya_a_dist_,
                self.lya_epsilon_,
            ) = self._build_a(self.S_, reuse=True, seeds=self.lya_ga_target_seeds)
            self.lya_log_pis_ = self.lya_a_dist_.log_prob(
                self.lya_a_
            )  # Gaussian actor action log_probability
            self.lya_l_ = self._build_l(
                self.S_, self.lya_a_, reuse=True, seed=self.lc_seed,
            )

            ###########################################
            # Create Loss functions and optimizers ####
            ###########################################

            # Lyapunov constraint function
            self.l_delta = tf.reduce_mean(
                input_tensor=(self.lya_l_ - self.l + ALPHA_3 * self.R)
            )

            # Lagrance multiplier loss functions and optimizers graphs
            self.labda_loss = -tf.reduce_mean(
                input_tensor=(self.log_labda * self.l_delta)
            )
            self.alpha_loss = -tf.reduce_mean(
                input_tensor=(
                    self.log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
                )
            )

            # Create optimizers

            # Alpha optimizer graph
            self.alpha_opt = tf.compat.v1.train.AdamOptimizer(self.LR_A)
            self.alpha_grads = self.alpha_opt.compute_gradients(
                self.alpha_loss, var_list=self.log_alpha
            )
            self.alpha_train = self.alpha_opt.apply_gradients(self.alpha_grads)

            # Lambda optimizer graph
            self.lambda_opt = tf.compat.v1.train.AdamOptimizer(self.LR_lag)
            self.lambda_grads = self.lambda_opt.compute_gradients(
                self.labda_loss, var_list=self.log_labda
            )
            self.lambda_train = self.lambda_opt.apply_gradients(self.lambda_grads)

            # Actor optimizer graph
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(
                input_tensor=log_pis
            )
            self.a_loss = a_loss
            self.a_opt = tf.compat.v1.train.AdamOptimizer(self.LR_A)
            self.a_grads = self.a_opt.compute_gradients(
                self.a_loss, var_list=self.a_params
            )
            self.a_train = self.a_opt.apply_gradients(self.a_grads)

            # Create Lyapunov Critic loss function and optimizer
            # NOTE: The control dependency makes sure the target networks are updated
            # first
            with tf.control_dependencies(target_update):

                # Lyapunov candidate constraint function graph
                self.l_target = self.R + GAMMA * (1 - self.terminal) * tf.stop_gradient(
                    self.l_
                )

                self.l_error = tf.compat.v1.losses.mean_squared_error(
                    labels=self.l_target, predictions=self.l
                )

                # New splitted optimizer
                self.l_opt = tf.compat.v1.train.AdamOptimizer(self.LR_L)
                self.l_grads = self.l_opt.compute_gradients(
                    self.l_error, var_list=self.l_params
                )
                self.l_train = self.l_opt.apply_gradients(self.l_grads)

            # Initialize variables, create saver and diagnostics graph
            self.entropy = tf.reduce_mean(input_tensor=-self.log_pis)
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            self.diagnostics = [
                self.l_delta,
                self.labda,
                self.alpha,
                self.log_labda,
                self.log_alpha,
                self.labda_loss,
                self.alpha_loss,
                self.l_target,
                self.l_error,
                self.a_loss,
                self.entropy,
                self.l,
                self.l_,
                self.lya_l_,
                self.a,
                self.a_,
                self.lya_a_,
                self.lambda_grads,
                self.alpha_grads,
                self.a_grads,
                self.l_grads,
            ]

            # Concatentate optimizer graphs
            self.opt = [self.l_train, self.lambda_train, self.a_train, self.alpha_train]

    def _build_a(
        self,
        s,
        name="gaussian_actor",
        reuse=None,
        custom_getter=None,
        seeds=[None, None],
    ):
        """Setup SquashedGaussianActor Graph.

        Args:
            s (tf.Tensor): [description]

            name (str, optional): Network name. Defaults to "actor".

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

            seeds (list, optional): The random seeds used for the weight initialization
                and the sampling ([weights_seed, sampling_seed]). Defaults to
                [None, None]

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create weight initializer
        initializer = tf.keras.initializers.GlorotUniform(seed=seeds[0])

        # Create graph
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer sizes
            n1 = self.network_structure["actor"][0]
            n2 = self.network_structure["actor"][1]

            # Create actor hidden/ output layers
            net_0 = tf.compat.v1.layers.dense(
                s,
                n1,
                activation=tf.nn.relu,
                name="l1",
                trainable=trainable,
                kernel_initializer=initializer,
            )  # 原始是30
            net_1 = tf.compat.v1.layers.dense(
                net_0,
                n2,
                activation=tf.nn.relu,
                name="l2",
                trainable=trainable,
                kernel_initializer=initializer,
            )  # 原始是30
            mu = tf.compat.v1.layers.dense(
                net_1,
                self.a_dim,
                activation=None,
                name="mu",
                trainable=trainable,
                kernel_initializer=initializer,
            )
            log_sigma = tf.compat.v1.layers.dense(
                net_1,
                self.a_dim,
                activation=None,
                name="log_sigma",
                trainable=trainable,
                kernel_initializer=initializer,
            )
            log_sigma = tf.clip_by_value(log_sigma, *LOG_SIGMA_MIN_MAX)

            # Calculate log probability standard deviation
            sigma = tf.exp(log_sigma)

            # Create bijectors (Used in the reparameterization trick)
            squash_bijector = SquashBijector()
            affine_bijector = tfp.bijectors.Shift(mu)(tfp.bijectors.Scale(sigma))

            # Sample from the normal distribution and calculate the action
            batch_size = tf.shape(input=s)[0]
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
            )
            epsilon = base_distribution.sample(batch_size, seed=seeds[1])
            raw_action = affine_bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            # Transform distribution back to the original policy distribution
            reparm_trick_bijector = tfp.bijectors.Chain(
                (squash_bijector, affine_bijector)
            )
            distribution = tfp.distributions.TransformedDistribution(
                distribution=base_distribution, bijector=reparm_trick_bijector
            )
            clipped_mu = squash_bijector.forward(mu)

        # Return network output graphs
        return clipped_a, clipped_mu, distribution, epsilon

    def _build_l(
        self, s, a, name="lyapunov_critic", reuse=None, custom_getter=None, seed=None,
    ):
        """Setup lyapunov critic graph.

        Args:
            s (tf.Tensor): Tensor of observations.

            a (tf.Tensor): Tensor with actions.

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

            seed (int, optional): The seed used for the weight initialization. Defaults
                to None.


        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create weight initializer
        initializer = tf.keras.initializers.GlorotUniform(seed=seed)

        # Create graph
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer size
            n1 = self.network_structure["critic"][0]

            # Create actor hidden/ output layers
            layers = []
            w1_s = tf.compat.v1.get_variable(
                "w1_s", [self.s_dim, n1], trainable=trainable, initializer=initializer,
            )
            w1_a = tf.compat.v1.get_variable(
                "w1_a", [self.a_dim, n1], trainable=trainable, initializer=initializer,
            )
            b1 = tf.compat.v1.get_variable(
                "b1", [1, n1], trainable=trainable, initializer=tf.zeros_initializer()
            )
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            layers.append(net_0)
            for i in range(1, len(self.network_structure["critic"])):
                n = self.network_structure["critic"][i]
                layers.append(
                    tf.compat.v1.layers.dense(
                        layers[i - 1],
                        n,
                        activation=tf.nn.relu,
                        name="l" + str(i + 1),
                        trainable=trainable,
                        kernel_initializer=initializer,
                    )
                )

            # Return lyapunov critic object
            return tf.expand_dims(
                tf.reduce_sum(input_tensor=tf.square(layers[-1]), axis=1), axis=1
            )  # L(s,a)

    def learn(self, LR_A, LR_L, LR_lag, batch):
        """Runs the SGD to update all the optimize parameters.

        Args:
            LR_A (float): Current actor learning rate.
            LR_L (float): Lyapunov critic learning rate.
            LR_lag (float): Lyapunov constraint langrance multiplier learning rate.
            batch (numpy.ndarray): The batch of experiences.

        Returns:
            Tuple: Tuple with diagnostics variables of the SGD.
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Fill optimizer variable feed critic
        feed_dict = {
            self.a_input: ba,
            self.S: bs,
            self.S_: bs_,
            self.R: br,
            self.terminal: bterminal,
            self.LR_A: LR_A,
            self.LR_L: LR_L,
            self.LR_lag: LR_lag,
        }

        # Run optimization and return diagnostics
        return self.sess.run([self.opt, self.diagnostics], feed_dict)[1]


####################################################
# Main function ####################################
####################################################
if __name__ == "__main__":

    # Create the Lyapunov Actor Critic agent
    policy = LAC()

    # Retrieve initial network weights
    (
        ga_weights_biases,
        ga_target_weights_biases,
        lc_weights_biases,
        lc_target_weights_biases,
    ) = retrieve_weights_biases()

    # Create dummy input
    # NOTE: Explicit seeding because of the difference between eager and graph mode
    tf.random.set_seed(0)
    s_tmp = tf.random.uniform((BATCH_SIZE, policy.s_dim), seed=0)
    a_tmp = tf.random.uniform((BATCH_SIZE, policy.a_dim), seed=1)
    r_tmp = tf.random.uniform((BATCH_SIZE, 1), seed=2)
    terminal_tmp = tf.cast(
        tf.random.uniform((BATCH_SIZE, 1), minval=0, maxval=2, dtype=tf.int32, seed=3),
        tf.float32,
    )
    s_target_tmp = tf.random.uniform((BATCH_SIZE, policy.s_dim), seed=4)
    batch = {
        "s": policy.sess.run(s_tmp),
        "a": policy.sess.run(a_tmp),
        "r": policy.sess.run(r_tmp),
        "terminal": policy.sess.run(terminal_tmp),
        "s_": policy.sess.run(s_target_tmp),
    }

    ################################################
    # Validate actor grads #########################
    ################################################

    # Create Actor Loss variables
    log_labda = tf.compat.v1.get_variable(
        "lambda", None, tf.float32, initializer=tf.math.log(LAMBDA)
    )
    log_alpha = tf.compat.v1.get_variable(
        "alpha", None, tf.float32, initializer=tf.math.log(ALPHA)
    )
    labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
    alpha = tf.exp(log_alpha)
    policy.sess.run(tf.compat.v1.global_variables_initializer())

    # Compute Lyapunov difference
    # NOTE: This is similar to the Q backup (Q_- Q + alpha * R) but now while the agent
    # tries to satisfy the the lyapunov stability constraint.
    l_delta = tf.reduce_mean(
        input_tensor=(policy.lya_l_ - policy.l + ALPHA_3 * policy.R)
    )

    # Compute actor loss
    # NOTE: Scale by 500 to make effects more prevalent.
    a_loss = GRAD_SCALE_FACTOR * (
        labda * l_delta + alpha * tf.reduce_mean(input_tensor=policy.log_pis)
    )

    # Create diagnostics retrieval graph
    a_diagnostics = [
        labda,
        alpha,
        l_delta,
        a_loss,
        policy.log_pis,
        policy.lya_l_,
        policy.l,
    ]

    # Compute actor gradients
    a_grads_graph = policy.a_opt.compute_gradients(a_loss, var_list=policy.a_params)
    (a_grads, a_diagnostics) = policy.sess.run(
        [a_grads_graph, a_diagnostics],
        feed_dict={
            policy.a_input: batch["a"],
            policy.S_: batch["s_"],
            policy.S: batch["s"],
            policy.R: batch["r"],
        },
    )
    a_grads_unpacked = [
        grads[0] for grads in a_grads
    ]  # Unpack gradients for easy comparison

    # Unpack diagnostics
    (labda, alpha, l_delta, a_loss, log_pis, lya_l_, l) = a_diagnostics

    # Print gradients
    print("\n==GAUSSIAN ACTOR GRADIENTS==")
    print("grad/l1/weights:")
    print(a_grads_unpacked[0])
    print("\ngrad/l1/bias:")
    print(a_grads_unpacked[1])
    print("\ngrad/l2/weights:")
    print(a_grads_unpacked[2])
    print("\ngrad/l2/bias:")
    print(a_grads_unpacked[3])
    print("\ngrad/mu/weights:")
    print(a_grads_unpacked[4])
    print("\ngrad/mu/bias:")
    print(a_grads_unpacked[5])
    print("\ngrad/log_sigma/weights:")
    print(a_grads_unpacked[6])
    print("\ngrad/log_sigma/bias:")
    print(a_grads_unpacked[7])

    ################################################
    # Validate critic grads ########################
    ################################################

    # Compute lyapunov Critic error
    # NOTE: Scale by 500 to make effects more prevalent.
    l_error = GRAD_SCALE_FACTOR * tf.compat.v1.losses.mean_squared_error(
        labels=policy.l_target, predictions=policy.l
    )

    # Create diagnostics retrieval graph
    l_diagnostics = [
        policy.l_target,
        l_error,
        policy.l,
        policy.l_,
    ]

    # Compute actor gradients
    l_grads_graph = policy.a_opt.compute_gradients(l_error, var_list=policy.l_params)
    (l_grads, l_diagnostics) = policy.sess.run(
        [l_grads_graph, l_diagnostics],
        feed_dict={
            policy.a_input: batch["a"],
            policy.S: batch["s"],
            policy.S_: batch["s_"],
            policy.terminal: batch["terminal"],
            policy.R: batch["r"],
        },
    )
    l_grads_unpacked = [
        grads[0] for grads in l_grads
    ]  # Unpack gradients for easy comparison

    # Unpack diagnostics
    (l_target, l_error, l, l_) = l_diagnostics

    # Print gradients
    print("\n==LYAPUNOV CRITIC GRADIENTS==")
    print("grad/l1/w1_s:")
    print(l_grads_unpacked[0])
    print("\ngrad/l1/w1_a:")
    print(l_grads_unpacked[1])
    print("\ngrad/l1/b1:")
    print(l_grads_unpacked[2])
    print("\ngrad/l2/weights:")
    print(l_grads_unpacked[3])
    print("\ngrad/l2/bias:")
    print(l_grads_unpacked[4])

    # End of the script
    print("End")
