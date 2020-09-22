"""Minimal working version of the LAC algorithm script.
"""

import time
from collections import deque
import os
import sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector
from utils import evaluate_training_rollouts
import logger
from pool import Pool

###############################################
# Script settings #############################
###############################################

# Environment parameters
ENV_NAME = "Ex3_EKF"  # The gym environment you want to train in

# Main training loop parameters
TRAIN_PARAMS = {
    "episodes": int(2e4),  # The number of episodes you want to perform
    "num_of_training_paths": 10,  # Number of training rollouts stored for analysis
    "evaluation_frequency": 2048,  # After how many steps the performance is evaluated
    "num_of_evaluation_paths": 10,  # number of rollouts for evaluation
    "num_of_trials": 1,  # number of random seed
    "start_of_trial": 0,  # The start number of the rollouts (used during model save)
}

# Network structure
NETWORK_STRUCTURE = {"critic": [128, 128], "actor": [64, 64]}

# Learning algorithm parameters
ALG_PARAMS = {
    "memory_capacity": int(1e6),  # The max replay buffer size
    "min_memory_size": 1000,  # The minimum replay buffer size before STG starts
    "batch_size": 256,  # The SGD batch size
    "labda": 1.0,  # Initial value for the lyapunov constraint lagrance multiplier
    "alpha": 1.0,  # The initial value for the entropy lagrance multiplier
    "alpha3": 0.2,  # The value of the stability condition multiplier
    "tau": 5e-3,  # Decay rate used in the polyak averaging
    "lr_a": 1e-4,  # The actor learning rate
    "lr_l": 3e-4,  # The lyapunov critic
    "gamma": 0.9,  # Discount factor
    "steps_per_cycle": 100,  # The number of steps after which the model is trained
    "train_per_cycle": 80,  # How many times SGD is called during the training
    "adaptive_alpha": True,  # Enables automatic entropy temperature tuning
    "target_entropy": None,  # Set alpha target entropy, when None == -(action_dim)
}

# Environment parameters
ENVS_PARAMS = {
    "oscillator": {
        "max_ep_steps": 800,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex3_EKF": {
        "max_ep_steps": 500,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
    "Ex4_EKF": {
        "max_ep_steps": 100,
        "max_global_steps": TRAIN_PARAMS["episodes"],
        "max_episodes": int(1e6),
        "eval_render": False,
    },
}

# Check if specified environment is valid
if ENV_NAME in ENVS_PARAMS.keys():
    ENV_PARAMS = ENVS_PARAMS[ENV_NAME]
else:
    print(
        f"Environmen {ENV_NAME} does not exist yet. Please specify a valid environment "
        "and try again."
    )
    sys.exit(0)

# Other paramters
TARGET_ENTROPY = None  # If None target entropy == -(action space dimension)
LOG_SIGMA_MIN_MAX = (-20, 2)  # Range of log std coming out of the GA network
SCALE_lambda_MIN_MAX = (0, 1)  # Range of lambda lagrance multiplier
# RANDOM_SEED = None  # The random seed you want to use. Use None to disable
RANDOM_SEED = 0  # The random seed you want to use. Use None to disable

# Set random seed to get comparable results for each run
if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


###############################################
# Helper functions ############################
###############################################
def get_env_from_name(name):
    """Initializes the gym environment with the given name

    Args:
        name (str): The name of the gym environment you want to initialize.

    Returns:
        gym.Env: The gym environment.
    """
    if name == "oscillator":
        from envs.oscillator import oscillator as env

        env = env()
        env = env.unwrapped
    elif name == "Ex3_EKF":
        from envs.Ex3_EKF import Ex3_EKF as env

        env = env()
        env = env.unwrapped
    elif name == "Ex4_EKF":
        from envs.Ex4_EKF import Ex4_EKF as env

        env = env()
        env = env.unwrapped
    if RANDOM_SEED:
        # env.seed(RANDOM_SEED)
        env.seed(None)
    return env


def training_evaluation(env, policy):
    """Evaluates the performance of the current policy in
    several rollouts.

    Args:
        env (gym.Env): The gym environment you want to use.
        policy (object): The current policy.

    Returns:
        [type]: [description]
    """
    # Retrieve action space bounds from env
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Training setting
    total_cost = []
    death_rates = []
    episode_length = []
    die_count = 0
    seed_average_cost = []

    # Perform roolouts to evaluate performance
    for i in range(TRAIN_PARAMS["num_of_evaluation_paths"]):
        cost = 0
        s = env.reset()
        for j in range(ENV_PARAMS["max_ep_steps"]):
            if ENV_PARAMS["eval_render"]:
                env.render()
            a = policy.choose_action(s, True)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2
            s_, r, done, _ = env.step(action)
            cost += r
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            s = s_
            if done:
                seed_average_cost.append(cost)
                episode_length.append(j)
                if j < ENV_PARAMS["max_ep_steps"] - 1:
                    die_count += 1
                break

    # Save evaluation results
    total_cost.append(np.mean(seed_average_cost))
    total_cost_mean = np.average(total_cost)
    average_length = np.average(episode_length)

    # Return evaluation results
    diagnostic = {
        "return": total_cost_mean,
        "average_length": average_length,
    }
    return diagnostic


# class Pool(object):
#     """Memory buffer class."""

#     def __init__(
#         self,
#         s_dim,
#         a_dim,
#         memory_capacity,
#         store_last_n_paths,
#         min_memory_size,
#     ):
#         """Initialize memory buffer object.

#         Args:
#             variant (dict): Dictionary containing all the required memory buffer
#             parameters.
#         """
#         self.memory_capacity = memory_capacity
#         self.paths = deque(maxlen=store_last_n_paths)
#         self.reset()
#         self.memory = {
#             "s": np.zeros([1, s_dim]),
#             "a": np.zeros([1, a_dim]),
#             "r": np.zeros([1, 1]),
#             "terminal": np.zeros([1, 1]),
#             "s_": np.zeros([1, s_dim]),
#         }
#         self.memory_pointer = 0
#         self.min_memory_size = min_memory_size

#     def reset(self):
#         """Reset memory buffer.
#         """
#         self.current_path = {
#             "s": [],
#             "a": [],
#             "r": [],
#             "terminal": [],
#             "s_": [],
#         }

#     def store(self, s, a, d, raw_d, r, terminal, s_):
#         """Store experience tuple.

#         Args:
#             s (numpy.ndarray): State.
#             a (numpy.ndarray): Action.
#             r (numpy.ndarray): Reward.
#             terminal (numpy.ndarray): Whether the terminal state was reached.
#             s_ (numpy.ndarray): Next state.

#         Returns:
#             int: The current memory buffer size.
#         """

#         # Store experience in memory buffer
#         transition = {
#             "s": s,
#             "a": a,
#             "r": np.array([r]),
#             "terminal": np.array([terminal]),
#             "s_": s_,
#         }
#         if len(self.current_path["s"]) < 1:
#             for key in transition.keys():
#                 self.current_path[key] = transition[key][np.newaxis, :]
#         else:
#             for key in transition.keys():
#                 self.current_path[key] = np.concatenate(
#                     (self.current_path[key], transition[key][np.newaxis, :])
#                 )
#         if terminal == 1.0:
#             for key in self.current_path.keys():
#                 self.memory[key] = np.concatenate(
#                     (self.memory[key], self.current_path[key]), axis=0
#                 )
#             self.paths.appendleft(self.current_path)
#             self.reset()
#             self.memory_pointer = len(self.memory["s"])

#         # Return current memory buffer size
#         return self.memory_pointer

#     def sample(self, batch_size):
#         """Sample from memory buffer.

#         Args:
#             batch_size (int): The memory buffer sample size.

#         Returns:
#             numpy.ndarray: The batch of experiences.
#         """
#         if self.memory_pointer < self.min_memory_size:
#             return None
#         else:

#             # Sample a random batch of experiences
#             indices = np.random.choice(
#                 min(self.memory_pointer, self.memory_capacity) - 1,
#                 size=batch_size,
#                 replace=False,
#             ) + max(1, 1 + self.memory_pointer - self.memory_capacity,) * np.ones(
#                 [batch_size], np.int
#             )
#             batch = {}
#             for key in self.memory.keys():
#                 if "s" in key:
#                     sample = self.memory[key][indices]
#                     batch.update({key: sample})
#                 else:
#                     batch.update({key: self.memory[key][indices]})
#             return batch


###############################################
# LAC algorithm class #########################
###############################################
class LAC(object):
    """The lyapunov actor critic.

    """

    def __init__(
        self, a_dim, s_dim,
    ):
        """Initiate object state.

        Args:
            a_dim (int): Action space dimension.
            s_dim (int): Observation space dimension.
        """

        # Save action and observation space as members
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Set algorithm parameters as class objects
        self.network_structure = NETWORK_STRUCTURE

        # Determine target entropy
        if TARGET_ENTROPY is None:
            self.target_entropy = -self.a_dim  # lower bound of the policy entropy
        else:
            self.target_entropy = TARGET_ENTROPY

        # Create tensorflow session
        self.sess = tf.Session()

        # Create networks, optimizers and variables inside the Actor scope
        with tf.variable_scope("Actor"):

            # Create observations placeholders
            self.S = tf.placeholder(tf.float32, [None, self.s_dim], "s")
            self.S_ = tf.placeholder(tf.float32, [None, self.s_dim], "s_")
            self.a_input = tf.placeholder(tf.float32, [None, self.a_dim], "a_input")
            self.a_input_ = tf.placeholder(tf.float32, [None, self.a_dim], "a_input_")
            self.R = tf.placeholder(tf.float32, [None, 1], "r")
            self.terminal = tf.placeholder(tf.float32, [None, 1], "terminal")

            # Create Learning rate placeholders
            self.LR_A = tf.placeholder(tf.float32, None, "LR_A")
            self.LR_lag = tf.placeholder(tf.float32, None, "LR_lag")
            self.LR_L = tf.placeholder(tf.float32, None, "LR_L")

            # Create lagrance multiplier placeholders
            log_labda = tf.get_variable(
                "lambda", None, tf.float32, initializer=tf.log(ALG_PARAMS["labda"])
            )
            log_alpha = tf.get_variable(
                "alpha", None, tf.float32, initializer=tf.log(ALG_PARAMS["alpha"])
            )
            self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(log_alpha)

            ###########################################
            # Create Networks #########################
            ###########################################

            # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
            self.a, self.deterministic_a, self.a_dist = self._build_a(self.S,)
            self.l = self._build_l(self.S, self.a_input)
            self.log_pis = log_pis = self.a_dist.log_prob(
                self.a
            )  # Gaussian actor action log_probability
            self.prob = tf.reduce_mean(
                self.a_dist.prob(self.a)
            )  # Gaussian actor action probability

            # Retrieve GA and LC network parameters
            a_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/actor"
            )
            l_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/Lyapunov"
            )

            # Create EMA target network update policy (Soft replacement)
            ema = tf.train.ExponentialMovingAverage(decay=1 - ALG_PARAMS["tau"])

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [
                ema.apply(a_params),
                ema.apply(l_params),
            ]

            # Create GA and LC target networks
            # Don't get optimized but get updated according to the EMA of the main
            # networks
            a_, _, a_dist_ = self._build_a(
                self.S_, reuse=True, custom_getter=ema_getter
            )
            l_ = self._build_l(self.S_, a_, reuse=True, custom_getter=ema_getter)

            # Create Networks for the (fixed) lyapunov temperature boundary
            # NOTE: Used as a minimum lambda constraint boundary
            lya_a_, _, _ = self._build_a(self.S_, reuse=True)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            ###########################################
            # Create Loss functions and optimizers ####
            ###########################################

            # Lyapunov constraint function
            self.l_delta = tf.reduce_mean(
                self.l_ - self.l + (ALG_PARAMS["alpha3"]) * self.R
            )

            # Lagrance multiplier loss functions and optimizers graphs
            labda_loss = -tf.reduce_mean(log_labda * self.l_delta)
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
            )
            self.alpha_train = tf.train.AdamOptimizer(self.LR_A).minimize(
                alpha_loss, var_list=log_alpha
            )
            self.lambda_train = tf.train.AdamOptimizer(self.LR_lag).minimize(
                labda_loss, var_list=log_labda
            )

            # Actor loss and optimizer graph
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(log_pis)
            self.a_loss = a_loss  # FIXME: IS this needed?
            self.a_train = tf.train.AdamOptimizer(self.LR_A).minimize(
                a_loss, var_list=a_params
            )

            # Create Lyapunov Critic loss function and optimizer
            # NOTE: The control dependency makes sure the target networks are updated \
            # first
            with tf.control_dependencies(target_update):

                # Lyapunov candidate constraint function graph
                l_target = self.R + ALG_PARAMS["gamma"] * (
                    1 - self.terminal
                ) * tf.stop_gradient(l_)

                self.l_error = tf.losses.mean_squared_error(
                    labels=l_target, predictions=self.l
                )
                self.l_train = tf.train.AdamOptimizer(self.LR_L).minimize(
                    self.l_error, var_list=l_params
                )

            # Initialize variables, create saver and diagnostics graph
            self.sess.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            self.diagnostics = [
                self.labda,
                self.alpha,
                self.l_error,
                tf.reduce_mean(-self.log_pis),
                self.a_loss,
                l_target,
                labda_loss,
                self.l_delta,
                log_labda,
                self.l_,
                self.l,
                self.R,
            ]

            # Create optimizer array
            self.opt = [self.l_train, self.lambda_train, self.a_train]
            if ALG_PARAMS["adaptive_alpha"]:
                self.opt.append(self.alpha_train)

    def choose_action(self, s, evaluation=False):
        """Returns the current action of the policy.

        Args:
            s (np.numpy): The current state.
            evaluation (bool, optional): Whether to return a deterministic action.
            Defaults to False.

        Returns:
            np.numpy: The current action.
        """
        if evaluation is True:
            try:
                return self.sess.run(self.deterministic_a, {self.S: s[np.newaxis, :]})[
                    0
                ]
            except ValueError:
                return
        else:
            return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self, LR_A, LR_L, LR_lag, batch):
        """Runs the SGD to update all the optimizable parameters.

        Args:
            LR_A (float): Current actor learning rate.
            LR_L (float): Lyapunov critic learning rate.
            LR_lag (float): Lyapunov constraint langrance multiplier learning rate.
            batch (numpy.ndarray): The batch of experiences.

        Returns:
            [type]: [description]
        """

        # Retrieve state, action and reward from the batch
        bs = batch["s"]  # state
        ba = batch["a"]  # action
        br = batch["r"]  # reward
        bterminal = batch["terminal"]
        bs_ = batch["s_"]  # next state

        # Fill optimizer variable feed cict
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

        # Run optimization
        self.sess.run(self.opt, feed_dict)

        # Retrieve diagnostic variables from the optimization
        (
            labda,
            alpha,
            l_error,
            entropy,
            a_loss,
            l_target,
            labda_loss,
            l_delta,
            log_labda,
            l_,
            l,
            R,
        ) = self.sess.run(self.diagnostics, feed_dict)

        # Return optimization results
        return labda, alpha, l_error, entropy, a_loss

    def _build_a(self, s, name="actor", reuse=None, custom_getter=None):
        """Setup SquashedGaussianActor Graph.

        Args:
            s (tf.Tensor): [description]

            name (str, optional): Network name. Defaults to "actor".

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create graph
        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

            # Retrieve hidden layer sizes
            n1 = self.network_structure["actor"][0]
            n2 = self.network_structure["actor"][1]

            # Create actor hidden/ output layers
            net_0 = tf.layers.dense(
                s, n1, activation=tf.nn.relu, name="l1", trainable=trainable
            )  # 原始是30
            net_1 = tf.layers.dense(
                net_0, n2, activation=tf.nn.relu, name="l4", trainable=trainable
            )  # 原始是30
            mu = tf.layers.dense(
                net_1, self.a_dim, activation=None, name="a", trainable=trainable
            )
            log_sigma = tf.layers.dense(net_1, self.a_dim, None, trainable=trainable)
            log_sigma = tf.clip_by_value(log_sigma, *LOG_SIGMA_MIN_MAX)

            # Calculate log probability standard deviation
            sigma = tf.exp(log_sigma)

            # Create bijectors (Used in the reparameterization trick)
            squash_bijector = SquashBijector()
            affine_bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)

            # Sample from the normal distribution and calculate the action
            batch_size = tf.shape(s)[0]
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self.a_dim), scale_diag=tf.ones(self.a_dim)
            )
            epsilon = base_distribution.sample(batch_size)
            raw_action = affine_bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            # Transform distribution back to the original policy distribution
            reparm_trick_bijector = tfp.bijectors.Chain(
                (squash_bijector, affine_bijector)
            )
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                distribution=base_distribution, bijector=reparm_trick_bijector
            )

            clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution

    def _build_l(self, s, a, reuse=None, custom_getter=None):
        """Setup lyapunov critic graph.

        Args:
            s (tf.Tensor): Tensor of observations.

            a (tf.Tensor): Tensor with actions.

            reuse (Bool, optional): Whether the network has to be trainable. Defaults
                to None.

            custom_getter (object, optional): Overloads variable creation process.
                Defaults to None.

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Set trainability
        trainable = True if reuse is None else False

        # Create graph
        with tf.variable_scope("Lyapunov", reuse=reuse, custom_getter=custom_getter):

            # Retrieve hidden layer size
            n1 = self.network_structure["critic"][0]

            # Create actor hidden/ output layers
            layers = []
            w1_s = tf.get_variable("w1_s", [self.s_dim, n1], trainable=trainable)
            w1_a = tf.get_variable("w1_a", [self.a_dim, n1], trainable=trainable)
            b1 = tf.get_variable("b1", [1, n1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            layers.append(net_0)
            for i in range(1, len(self.network_structure["critic"])):
                n = self.network_structure["critic"][i]
                layers.append(
                    tf.layers.dense(
                        layers[i - 1],
                        n,
                        activation=tf.nn.relu,
                        name="l" + str(i + 1),
                        trainable=trainable,
                    )
                )

            # Return lyapunov critic object
            return tf.expand_dims(
                tf.reduce_sum(tf.square(layers[-1]), axis=1), axis=1
            )  # L(s,a)

    def save_result(self, path):
        """Save current policy.

        Args:
            path (str): The path where you want to save the policy.
        """

        save_path = self.saver.save(self.sess, path + "/policy/model.ckpt")
        print("Save to path: ", save_path)

    def restore(self, path):
        """Restore policy.

        Args:
            path (str): The path where you want to save the policy.

        Returns:
            bool: Boolean specifying whether the policy was loaded succesfully.
        """
        model_file = tf.train.latest_checkpoint(path + "/")
        if model_file is None:
            success_load = False
            return success_load
        self.saver.restore(self.sess, model_file)
        success_load = True
        return success_load


def train(log_dir):
    """Performs the agent traning.

    Args:
        log_dir (str): The directory in which the final model (policy) and the
        log data is saved.
    """

    # Create environment
    env = get_env_from_name(ENV_NAME)

    # Set initial learning rates
    lr_a, lr_l = (
        ALG_PARAMS["lr_a"],
        ALG_PARAMS["lr_l"],
    )
    lr_a_now = ALG_PARAMS["lr_a"]  # learning rate for actor, lambda and alpha
    lr_l_now = ALG_PARAMS["lr_l"]  # learning rate for lyapunov critic

    # Get observation and action space dimension and limits from the environment
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_upperbound = env.action_space.high
    a_lowerbound = env.action_space.low

    # Create the Lyapunov Actor Critic agent
    policy = LAC(a_dim, s_dim)

    pool_params = {
        "s_dim": s_dim,
        "a_dim": a_dim,
        "d_dim": 1,
        "store_last_n_paths": TRAIN_PARAMS["num_of_training_paths"],
        "memory_capacity": ALG_PARAMS["memory_capacity"],
        "min_memory_size": ALG_PARAMS["min_memory_size"],
    }
    pool = Pool(pool_params)
    # Create replay memory buffer
    # pool = Pool(
    #     s_dim=s_dim,
    #     a_dim=a_dim,
    #     store_last_n_paths=TRAIN_PARAMS["num_of_training_paths"],
    #     memory_capacity=ALG_PARAMS["memory_capacity"],
    #     min_memory_size=ALG_PARAMS["min_memory_size"],
    # )

    # Training setting
    t1 = time.time()
    global_step = 0
    last_training_paths = deque(maxlen=TRAIN_PARAMS["num_of_training_paths"])
    training_started = False

    # Setup logger and log hyperparameters
    logger.configure(dir=log_dir, format_strs=["csv"])
    logger.logkv("tau", ALG_PARAMS["tau"])
    logger.logkv("alpha3", ALG_PARAMS["alpha3"])
    logger.logkv("batch_size", ALG_PARAMS["batch_size"])
    logger.logkv("target_entropy", policy.target_entropy)

    # Training loop
    for i in range(ENV_PARAMS["max_episodes"]):

        # Create variable to store information about the current path
        current_path = {
            "rewards": [],
            "a_loss": [],
            "alpha": [],
            "lambda": [],
            "lyapunov_error": [],
            "entropy": [],
        }

        # Stop training if max number of steps has been reached
        if global_step > ENV_PARAMS["max_global_steps"]:
            break

        # Reset environment
        s = env.reset()

        # Training Episode loop
        for j in range(ENV_PARAMS["max_ep_steps"]):

            # Render environment if requested
            if ENV_PARAMS["eval_render"]:
                env.render()

            # Retrieve (scaled) action based on the current policy
            a = policy.choose_action(s)
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2

            # Perform action in env
            s_, r, done, info = env.step(action)

            # Increment global setp count
            if training_started:
                global_step += 1

            # Stop episode if max_steps has been reached
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            terminal = 1.0 if done else 0.0

            # Store experience in replay buffer
            pool.store(s, a, np.zeros([1]), np.zeros([1]), r, terminal, s_)

            # Optimzize weights and parameters using STG
            if (
                pool.memory_pointer > ALG_PARAMS["min_memory_size"]
                and global_step % ALG_PARAMS["steps_per_cycle"] == 0
            ):
                training_started = True

                # Perform STG a set number of times (train per cycle)
                for _ in range(ALG_PARAMS["train_per_cycle"]):
                    batch = pool.sample(ALG_PARAMS["batch_size"])
                    labda, alpha, l_loss, entropy, a_loss = policy.learn(
                        lr_a_now, lr_l_now, lr_a, batch
                    )

            # Save path results
            if training_started:
                current_path["rewards"].append(r)
                current_path["lyapunov_error"].append(l_loss)
                current_path["alpha"].append(alpha)
                current_path["lambda"].append(labda)
                current_path["entropy"].append(entropy)
                current_path["a_loss"].append(a_loss)

            # Evalute the current performance and log results
            if (
                training_started
                and global_step % TRAIN_PARAMS["evaluation_frequency"] == 0
                and global_step > 0
            ):
                logger.logkv("total_timesteps", global_step)
                training_diagnostics = evaluate_training_rollouts(last_training_paths)
                if training_diagnostics is not None:
                    if TRAIN_PARAMS["num_of_evaluation_paths"] > 0:
                        eval_diagnostics = training_evaluation(env, policy)
                        [
                            logger.logkv(key, eval_diagnostics[key])
                            for key in eval_diagnostics.keys()
                        ]
                        training_diagnostics.pop("return")
                    [
                        logger.logkv(key, training_diagnostics[key])
                        for key in training_diagnostics.keys()
                    ]
                    logger.logkv("lr_a", lr_a_now)
                    logger.logkv("lr_l", lr_l_now)
                    string_to_print = ["time_step:", str(global_step), "|"]
                    if TRAIN_PARAMS["num_of_evaluation_paths"] > 0:
                        [
                            string_to_print.extend(
                                [key, ":", str(eval_diagnostics[key]), "|"]
                            )
                            for key in eval_diagnostics.keys()
                        ]
                    [
                        string_to_print.extend(
                            [key, ":", str(round(training_diagnostics[key], 2)), "|"]
                        )
                        for key in training_diagnostics.keys()
                    ]
                    print("".join(string_to_print))
                logger.dumpkvs()

            # Update state
            s = s_

            # Decay learning rate
            if done:
                if training_started:
                    last_training_paths.appendleft(current_path)
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                break

    # Save model and print Running time
    policy.save_result(log_dir)
    print("Running time: ", time.time() - t1)
    return


###############################################
# Main function ###############################
###############################################
if __name__ == "__main__":

    # Setup log path
    log_path = "/".join(["../log", ENV_NAME, "LAC" + time.strftime("%Y%m%d_%H%M"),])

    # Train several agents in the environment and save the results
    for i in range(
        TRAIN_PARAMS["start_of_trial"],
        TRAIN_PARAMS["start_of_trial"] + TRAIN_PARAMS["num_of_trials"],
    ):
        roll_out_log_path = log_path + "/" + str(i)
        print("logging to " + roll_out_log_path)
        train(roll_out_log_path)
        tf.reset_default_graph()
