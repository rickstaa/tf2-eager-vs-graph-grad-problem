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
    ga_weights_biases = sess.run(ga_vars)
    ga_target_weights_biases = sess.run(ga_target_vars)
    lc_weights_biases = sess.run(lc_vars)
    lc_target_weights_biases = sess.run(lc_target_vars)
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
# Used network functions ###########################
####################################################
def SquashedGaussianActorGraph(
    s, name="gaussian_actor", reuse=None, custom_getter=None, seeds=[None, None],
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
    with tf.compat.v1.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

        # Retrieve hidden layer sizes
        n1 = NETWORK_STRUCTURE["actor"][0]
        n2 = NETWORK_STRUCTURE["actor"][1]

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
            A_DIM,
            activation=None,
            name="mu",
            trainable=trainable,
            kernel_initializer=initializer,
        )
        log_sigma = tf.compat.v1.layers.dense(
            net_1,
            A_DIM,
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
            loc=tf.zeros(A_DIM), scale_diag=tf.ones(A_DIM)
        )
        epsilon = base_distribution.sample(batch_size, seed=seeds[1])
        raw_action = affine_bijector.forward(epsilon)
        clipped_a = squash_bijector.forward(raw_action)

        # Transform distribution back to the original policy distribution
        reparm_trick_bijector = tfp.bijectors.Chain((squash_bijector, affine_bijector))
        distribution = tfp.distributions.TransformedDistribution(
            distribution=base_distribution, bijector=reparm_trick_bijector
        )
        clipped_mu = squash_bijector.forward(mu)

    # Return network output graphs
    return clipped_a, clipped_mu, distribution, epsilon


def LyapunovCriticGraph(
    s, a, name="lyapunov_critic", reuse=None, custom_getter=None, seed=None,
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
    with tf.compat.v1.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

        # Retrieve hidden layer size
        n1 = NETWORK_STRUCTURE["critic"][0]

        # Create actor hidden/ output layers
        layers = []
        w1_s = tf.compat.v1.get_variable(
            "w1_s", [S_DIM, n1], trainable=trainable, initializer=initializer,
        )
        w1_a = tf.compat.v1.get_variable(
            "w1_a", [A_DIM, n1], trainable=trainable, initializer=initializer,
        )
        b1 = tf.compat.v1.get_variable(
            "b1", [1, n1], trainable=trainable, initializer=tf.zeros_initializer()
        )
        net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
        layers.append(net_0)
        for i in range(1, len(NETWORK_STRUCTURE["critic"])):
            n = NETWORK_STRUCTURE["critic"][i]
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


####################################################
# Main function ####################################
####################################################
if __name__ == "__main__":

    # Set algorithm parameters as class objects
    log_labda = tf.Variable(tf.math.log(LAMBDA), name="log_lambda")
    log_alpha = tf.Variable(tf.math.log(ALPHA), name="log_alpha")
    labda = tf.math.exp(log_labda)
    alpha = tf.math.exp(log_alpha)

    # Determine target entropy
    target_entropy = -A_DIM  # lower bound of the policy entropy

    # Create network seeds
    ga_seeds = [
        RANDOM_SEED,
        TFP_SEED_STREAM(),
    ]  # [weight init seed, sample seed]
    ga_target_seeds = [
        RANDOM_SEED + 1,
        TFP_SEED_STREAM(),
    ]  # [weight init seed, sample seed]
    lya_ga_target_seeds = [
        RANDOM_SEED,
        TFP_SEED_STREAM(),
    ]  # [weight init seed, sample seed]
    lc_seed = RANDOM_SEED + 2  # Weight init seed
    lc_target_seed = RANDOM_SEED + 3  # Weight init seed

    # Create tensorflow session
    sess = tf.compat.v1.Session()

    ###########################################
    # Create graphs ###########################
    ###########################################

    # Create variables, placeholders, networks and loss functions
    with tf.compat.v1.variable_scope("Actor"):

        # Create observations placeholders
        S = tf.compat.v1.placeholder(tf.float32, [None, S_DIM], "s")
        S_ = tf.compat.v1.placeholder(tf.float32, [None, S_DIM], "s_")
        a_input = tf.compat.v1.placeholder(tf.float32, [None, A_DIM], "a_input")
        R = tf.compat.v1.placeholder(tf.float32, [None, 1], "r")
        terminal = tf.compat.v1.placeholder(tf.float32, [None, 1], "terminal")

        # Create Learning rate placeholders
        LR_A = tf.compat.v1.placeholder(tf.float32, None, "LR_A")
        LR_lag = tf.compat.v1.placeholder(tf.float32, None, "LR_lag")
        LR_L = tf.compat.v1.placeholder(tf.float32, None, "LR_L")

        # Create lagrance multiplier placeholders
        log_labda = tf.compat.v1.get_variable(
            "lambda", None, tf.float32, initializer=tf.math.log(LAMBDA)
        )
        log_alpha = tf.compat.v1.get_variable(
            "alpha", None, tf.float32, initializer=tf.math.log(ALPHA)
        )
        labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
        alpha = tf.exp(log_alpha)

        ###########################################
        # Create Networks #########################
        ###########################################

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        a, _, a_dist, _ = SquashedGaussianActorGraph(S, seeds=ga_seeds)
        l = LyapunovCriticGraph(S, a_input, seed=lc_seed)
        log_pis = a_dist.log_prob(a)  # Gaussian actor action log_probability

        # Retrieve GA and LC network parameters
        a_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/gaussian_actor"
        )
        l_params = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/lyapunov_critic",
        )

        # Create EMA target network update policy (Soft replacement)
        ema = tf.train.ExponentialMovingAverage(decay=(POLYAK))

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [
            ema.apply(a_params),
            ema.apply(l_params),
        ]

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        a_, _, _, _ = SquashedGaussianActorGraph(
            S_, reuse=True, custom_getter=ema_getter, seeds=ga_target_seeds,
        )
        l_ = LyapunovCriticGraph(
            S_, a_, reuse=True, custom_getter=ema_getter, seed=lc_target_seed,
        )

        # Create Networks for the (fixed) lyapunov temperature boundary
        # DEBUG: This graph has the same parameters as the original gaussian actor
        # but now it receives the next state. This was needed as the target network
        # uses exponential moving average.
        (lya_a_, _, _, _,) = SquashedGaussianActorGraph(
            S_, reuse=True, seeds=lya_ga_target_seeds
        )
        lya_l_ = LyapunovCriticGraph(S_, lya_a_, reuse=True, seed=lc_seed)

        ###########################################
        # Create Loss functions and optimizers ####
        ###########################################

        # Lyapunov constraint function
        l_delta = tf.reduce_mean(input_tensor=(lya_l_ - l + ALPHA_3 * R))

        # Create optimizers

        # Actor optimizer graph
        a_loss = labda * l_delta + alpha * tf.reduce_mean(input_tensor=log_pis)
        a_opt = tf.compat.v1.train.AdamOptimizer(LR_A)
        a_grads = a_opt.compute_gradients(a_loss, var_list=a_params)

        # Create Lyapunov Critic loss function and optimizer
        # NOTE: The control dependency makes sure the target networks are updated
        # first
        with tf.control_dependencies(target_update):

            # Lyapunov candidate constraint function graph
            l_target = R + GAMMA * (1 - terminal) * tf.stop_gradient(l_)

            l_error = tf.compat.v1.losses.mean_squared_error(
                labels=l_target, predictions=l
            )

            # New splitted optimizer
            l_opt = tf.compat.v1.train.AdamOptimizer(LR_L)
            l_grads = l_opt.compute_gradients(l_error, var_list=l_params)

        # Initialize variables, create saver and diagnostics graph
        sess.run(tf.compat.v1.global_variables_initializer())

    ###########################################
    # Retrieve weights and create dummy batch #
    ###########################################

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
    s_tmp = tf.random.uniform((BATCH_SIZE, S_DIM), seed=0)
    a_tmp = tf.random.uniform((BATCH_SIZE, A_DIM), seed=1)
    r_tmp = tf.random.uniform((BATCH_SIZE, 1), seed=2)
    terminal_tmp = tf.cast(
        tf.random.uniform((BATCH_SIZE, 1), minval=0, maxval=2, dtype=tf.int32, seed=3),
        tf.float32,
    )
    s_target_tmp = tf.random.uniform((BATCH_SIZE, S_DIM), seed=4)
    batch = {
        "s": sess.run(s_tmp),
        "a": sess.run(a_tmp),
        "r": sess.run(r_tmp),
        "terminal": sess.run(terminal_tmp),
        "s_": sess.run(s_target_tmp),
    }

    ################################################
    # Validate actor grads #########################
    ################################################

    # Initialise variables
    sess.run(tf.compat.v1.global_variables_initializer())

    # Compute Lyapunov difference
    # NOTE: This is similar to the Q backup (Q_- Q + alpha * R) but now while the agent
    # tries to satisfy the the lyapunov stability constraint.
    l_delta = tf.reduce_mean(input_tensor=(lya_l_ - l + ALPHA_3 * R))

    # Compute actor loss
    # NOTE: Scale by factor to make effects more prevalent.
    a_loss = GRAD_SCALE_FACTOR * (
        labda * l_delta + alpha * tf.reduce_mean(input_tensor=log_pis)
    )

    # Create diagnostics retrieval graph
    a_diagnostics = [
        labda,
        alpha,
        l_delta,
        a_loss,
        log_pis,
        lya_l_,
        l,
    ]

    # Compute actor gradients
    a_grads_graph = a_opt.compute_gradients(a_loss, var_list=a_params)
    (a_grads_val, a_diagnostics) = sess.run(
        [a_grads_graph, a_diagnostics],
        feed_dict={a_input: batch["a"], S_: batch["s_"], S: batch["s"], R: batch["r"],},
    )
    a_grads_unpacked = [
        grads[0] for grads in a_grads_val
    ]  # Unpack gradients for easy comparison

    # Unpack diagnostics
    (
        labda_val,
        alpha_val,
        l_delta_val,
        a_loss_val,
        log_pis_val,
        lya_l_val,
        l_val,
    ) = a_diagnostics

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
    # NOTE: Scale by factor to make effects more prevalent.
    l_error = GRAD_SCALE_FACTOR * tf.compat.v1.losses.mean_squared_error(
        labels=l_target, predictions=l
    )

    # Create diagnostics retrieval graph
    l_diagnostics = [
        l_target,
        l_error,
        l,
        l_,
    ]

    # Compute actor gradients
    l_grads_graph = a_opt.compute_gradients(l_error, var_list=l_params)
    (l_grads_val, l_diagnostics) = sess.run(
        [l_grads_graph, l_diagnostics],
        feed_dict={
            a_input: batch["a"],
            S: batch["s"],
            S_: batch["s_"],
            terminal: batch["terminal"],
            R: batch["r"],
        },
    )
    l_grads_unpacked = [
        grads[0] for grads in l_grads_val
    ]  # Unpack gradients for easy comparison

    # Unpack diagnostics
    (l_target_val, l_error_Val, l_val, l_target_val) = l_diagnostics

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
