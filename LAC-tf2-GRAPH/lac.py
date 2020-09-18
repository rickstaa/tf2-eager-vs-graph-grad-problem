"""Minimal working version of the LAC algorithm script.
"""

import time
from collections import deque
import os
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from squash_bijector import SquashBijector
from utils import evaluate_training_rollouts, get_env_from_name, training_evaluation
import logger
from pool import Pool

###############################################
# Script settings #############################
###############################################
from variant import (
    USE_GPU,
    ENV_NAME,
    RANDOM_SEED,
    ENV_SEED,
    TRAIN_PARAMS,
    ALG_PARAMS,
    ENV_PARAMS,
    LOG_SIGMA_MIN_MAX,
    SCALE_lambda_MIN_MAX,
)

# Set random seed to get comparable results for each run
if RANDOM_SEED is not None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    np.random.seed(RANDOM_SEED)
    tf.compat.v1.reset_default_graph()
    tf.random.set_seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

# Disable eager
tf.compat.v1.disable_eager_execution()

# Disable GPU if requested
if not USE_GPU:
    tf.config.set_visible_devices([], "GPU")

# Tensorboard settings
USE_TB = True  # Whether you want to log to tensorboard
TB_FREQ = 4  # After how many episode we want to log to tensorboard
WRITE_W_B = False  # Whether you want to log the model weights and biases


###############################################
# LAC algorithm class #########################
###############################################
# TODO: Fix eval mode!
class LAC(object):
    """The lyapunov actor critic.

    """

    def __init__(self, a_dim, s_dim, log_dir="."):
        """Initiate object state.

        Args:
            a_dim (int): Action space dimension.
            s_dim (int): Observation space dimension.
        """

        # Save action and observation space as members
        self.a_dim = a_dim
        self.s_dim = s_dim

        # Set algorithm parameters as class objects
        self.network_structure = ALG_PARAMS["network_structure"]

        # Determine target entropy
        if ALG_PARAMS["target_entropy"] is None:
            self.target_entropy = -self.a_dim  # lower bound of the policy entropy
        else:
            self.target_entropy = ALG_PARAMS["target_entropy"]

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
            log_labda = tf.compat.v1.get_variable(
                "lambda", None, tf.float32, initializer=tf.math.log(ALG_PARAMS["labda"])
            )
            log_alpha = tf.compat.v1.get_variable(
                "alpha", None, tf.float32, initializer=tf.math.log(ALG_PARAMS["alpha"])
            )
            self.labda = tf.clip_by_value(tf.exp(log_labda), *SCALE_lambda_MIN_MAX)
            self.alpha = tf.exp(log_alpha)

            ###########################################
            # Create Networks #########################
            ###########################################

            # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
            self.a, self.deterministic_a, self.a_dist = self._build_a(self.S)
            self.l = self._build_l(self.S, self.a_input)
            self.log_pis = log_pis = self.a_dist.log_prob(
                self.a
            )  # Gaussian actor action log_probability

            # Retrieve GA and LC network parameters
            a_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/gaussian_actor"
            )
            l_params = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES,
                scope="Actor/lyapunov_critic",
            )

            # Create EMA target network update policy (Soft replacement)
            ema = tf.train.ExponentialMovingAverage(decay=(1 - ALG_PARAMS["tau"]))

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
            # DEBUG: This Network has the same parameters as the original gaussian actor but
            # now it receives the next state. This was needed as the target network
            # uses exponential moving average.
            # NOTE: Used as a minimum lambda constraint boundary
            lya_a_, _, _ = self._build_a(self.S_, reuse=True)
            self.l_ = self._build_l(self.S_, lya_a_, reuse=True)

            ###########################################
            # Create Loss functions and optimizers ####
            ###########################################

            # Lyapunov constraint function
            self.l_delta = tf.reduce_mean(
                input_tensor=(self.l_ - self.l + (ALG_PARAMS["alpha3"]) * self.R)
            )

            # Lagrance multiplier loss functions and optimizers graphs
            labda_loss = -tf.reduce_mean(input_tensor=(log_labda * self.l_delta))
            alpha_loss = -tf.reduce_mean(
                input_tensor=(
                    log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
                )
            )
            self.alpha_train = tf.compat.v1.train.AdamOptimizer(self.LR_A).minimize(
                alpha_loss, var_list=log_alpha
            )
            self.lambda_train = tf.compat.v1.train.AdamOptimizer(self.LR_lag).minimize(
                labda_loss, var_list=log_labda
            )

            # Actor loss and optimizer graph
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(
                input_tensor=log_pis
            )
            self.a_loss = a_loss  # FIXME: IS this needed?
            self.a_train = tf.compat.v1.train.AdamOptimizer(self.LR_A).minimize(
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

                self.l_error = tf.compat.v1.losses.mean_squared_error(
                    labels=l_target, predictions=self.l
                )
                self.l_train = tf.compat.v1.train.AdamOptimizer(self.LR_L).minimize(
                    self.l_error, var_list=l_params
                )

            # Initialize variables, create saver and diagnostics graph
            self.entropy = tf.reduce_mean(input_tensor=-self.log_pis)
            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()
            self.diagnostics = [
                self.labda,
                self.alpha,
                self.l_error,
                self.entropy,
                self.a_loss,
                l_target,
                alpha_loss,
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

        # Create summary writer
        if USE_TB:
            self.step = tf.Variable(0, dtype=tf.int64)
            self.sess.run(self.step.initializer)
            self.tb_writer = tf.compat.v1.summary.FileWriter(log_dir, self.sess.graph)

        # Retrieve weights names
        if WRITE_W_B:
            self._ga_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/gaussian_actor",
            )
            self._ga_target_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope="Actor/Actor/gaussian_actor",
            )
            self._lc_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES, scope="Actor/lyapunov_critic",
            )
            self._lc_target_vars = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                scope="Actor/Actor/lyapunov_critic",
            )
            self._ga_vars_names = [
                (item.name)
                .replace("Actor/gaussian_actor", "Ga")
                .replace("/kernel", "/weights")
                .replace("/bias", "/bias")
                .replace(":0", "")
                for item in self._ga_vars
            ]
            self._ga_target_vars_names = [
                (item.name)
                .replace("Actor/Actor/gaussian_actor", "Ga_")
                .replace("/kernel", "/weights",)
                .replace("/bias", "/bias")
                .replace("/ExponentialMovingAverage", "")
                .replace(":0", "")
                for item in self._ga_target_vars
                if "/Adam" not in item.name
            ]
            self._lc_vars_names = [
                (item.name)
                .replace("Actor/lyapunov_critic", "Lc")
                .replace("/kernel", "/weights")
                .replace("/bias", "/bias")
                .replace(":0", "")
                for item in self._lc_vars
            ]
            self._lc_target_vars_names = [
                (item.name)
                .replace("Actor/Actor/lyapunov_critic", "Lc_")
                .replace("/kernel", "/weights")
                .replace("/bias", "/bias")
                .replace("/ExponentialMovingAverage", "")
                .replace(":0", "")
                for item in self._lc_target_vars
                if "/Adam" not in item.name
            ]

            # Create weights/baises summary
            ga_sum = []
            for name, val in zip(self._ga_vars_names, self._ga_vars):  # GA
                ga_sum.append(tf.compat.v1.summary.histogram(name, val))
            ga_target_sum = []
            for name, val in zip(
                self._ga_target_vars_names, self._ga_target_vars
            ):  # GA
                ga_target_sum.append(tf.compat.v1.summary.histogram(name, val))
            lc_sum = []
            for name, val in zip(self._lc_vars_names, self._lc_vars):  # GA
                lc_sum.append(tf.compat.v1.summary.histogram(name, val))
            lc_target_sum = []
            for name, val in zip(
                self._lc_target_vars_names, self._lc_target_vars
            ):  # GA
                lc_target_sum.append(tf.compat.v1.summary.histogram(name, val))
            self.w_b_sum = tf.compat.v1.summary.merge(
                [ga_sum, ga_target_sum, lc_sum, lc_target_sum]
            )

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
            alpha_loss,
            labda_loss,
            l_delta,
            log_labda,
            l_,
            l,
            R,
        ) = self.sess.run(self.diagnostics, feed_dict)

        # Return optimization results
        return labda, alpha, l_error, entropy, a_loss

    def _build_a(self, s, name="gaussian_actor", reuse=None, custom_getter=None):
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
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer sizes
            n1 = self.network_structure["actor"][0]
            n2 = self.network_structure["actor"][1]

            # Create actor hidden/ output layers
            net_0 = tf.compat.v1.layers.dense(
                s, n1, activation=tf.nn.relu, name="l1", trainable=trainable
            )  # 原始是30
            net_1 = tf.compat.v1.layers.dense(
                net_0, n2, activation=tf.nn.relu, name="l2", trainable=trainable
            )  # 原始是30
            mu = tf.compat.v1.layers.dense(
                net_1, self.a_dim, activation=None, name="mu", trainable=trainable
            )
            log_sigma = tf.compat.v1.layers.dense(
                net_1,
                self.a_dim,
                activation=None,
                name="log_sigma",
                trainable=trainable,
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
            tfp_seed = tfp.util.SeedStream(RANDOM_SEED, salt="random_beta")
            epsilon = base_distribution.sample(batch_size, seed=tfp_seed())
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

        return clipped_a, clipped_mu, distribution

    def _build_l(self, s, a, name="lyapunov_critic", reuse=None, custom_getter=None):
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
        with tf.compat.v1.variable_scope(
            name, reuse=reuse, custom_getter=custom_getter
        ):

            # Retrieve hidden layer size
            n1 = self.network_structure["critic"][0]

            # Create actor hidden/ output layers
            layers = []
            w1_s = tf.compat.v1.get_variable(
                "w1_s", [self.s_dim, n1], trainable=trainable
            )
            w1_a = tf.compat.v1.get_variable(
                "w1_a", [self.a_dim, n1], trainable=trainable
            )
            # b1 = tf.compat.v1.get_variable("b1", [1, n1], trainable=trainable)
            b1 = tf.compat.v1.get_variable(
                "b1", [1, n1], trainable=trainable, initializer=tf.zeros_initializer()
            )  # DEBUG
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
                    )
                )

            # Return lyapunov critic object
            return tf.expand_dims(
                tf.reduce_sum(input_tensor=tf.square(layers[-1]), axis=1), axis=1
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
    env = get_env_from_name(ENV_NAME, ENV_SEED)

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
    policy = LAC(a_dim, s_dim, log_dir=log_dir)

    # Create replay memory buffer
    pool = Pool(
        s_dim=s_dim,
        a_dim=a_dim,
        store_last_n_paths=TRAIN_PARAMS["num_of_training_paths"],
        memory_capacity=ALG_PARAMS["memory_capacity"],
        min_memory_size=ALG_PARAMS["min_memory_size"],
    )

    # Training setting
    t1 = time.time()
    global_step = 0
    tb_step = 0
    last_training_paths = deque(maxlen=TRAIN_PARAMS["num_of_training_paths"])
    training_started = False

    # Create tensorboard variables
    tb_lr_a = tf.Variable(lr_a, dtype=tf.float32)
    tb_lr_l = tf.Variable(lr_l, dtype=tf.float32)
    tb_lr_lag = tf.Variable(lr_a, dtype=tf.float32)
    tb_ret = tf.Variable(0, dtype=tf.float32)
    tb_len = tf.Variable(0, dtype=tf.float32)
    tb_a_loss = tf.Variable(0, dtype=tf.float32)
    tb_lyapunov_error = tf.Variable(0, dtype=tf.float32)
    tb_entropy = tf.Variable(0, dtype=tf.float32)

    # Initialize tensorboard variables and create summaries
    if USE_TB:
        policy.sess.run(
            [
                tb_lr_a.initializer,
                tb_lr_l.initializer,
                tb_lr_lag.initializer,
                tb_ret.initializer,
                tb_len.initializer,
                tb_a_loss.initializer,
                tb_lyapunov_error.initializer,
                tb_entropy.initializer,
            ]
        )

        # Add tensorboard summaries
        main_sum = tf.compat.v1.summary.merge(
            [
                tf.compat.v1.summary.scalar("lr_a", tb_lr_a),
                tf.compat.v1.summary.scalar("lr_l", tb_lr_l),
                tf.compat.v1.summary.scalar("lr_lag", tb_lr_lag),
                tf.compat.v1.summary.scalar("alpha", policy.alpha),
                tf.compat.v1.summary.scalar("lambda", policy.labda),
            ]
        )
        other_sum = tf.compat.v1.summary.merge(
            [
                tf.compat.v1.summary.scalar("ep_ret", tb_ret),
                tf.compat.v1.summary.scalar("ep_length", tb_len),
                tf.compat.v1.summary.scalar("a_loss", tb_a_loss),
                tf.compat.v1.summary.scalar("lyapunov_error", tb_lyapunov_error),
                tf.compat.v1.summary.scalar("entropy", tb_entropy),
            ]
        )
        policy.tb_writer.add_summary(
            policy.sess.run(main_sum), policy.sess.run(policy.step)
        )
        if WRITE_W_B:
            policy.tb_writer.add_summary(
                policy.sess.run(policy.w_b_sum), policy.sess.run(policy.step),
            )
        policy.tb_writer.flush()  # Above summaries are known from the start

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
            # a = np.squeeze(np.random.uniform(low=-1.0, high=1.0, size=(1, 2)))  # DEBUG
            action = a_lowerbound + (a + 1.0) * (a_upperbound - a_lowerbound) / 2

            # Perform action in env
            s_, r, done, _ = env.step(action)

            # Increment global step count
            if training_started:
                global_step += 1

            # Stop episode if max_steps has been reached
            if j == ENV_PARAMS["max_ep_steps"] - 1:
                done = True
            terminal = 1.0 if done else 0.0

            # Store experience in replay buffer
            pool.store(s, a, r, terminal, s_)

            # Increment tensorboard step counter
            # NOTE: This was done differently from the global_step counter since
            # otherwise there were inconsistencies in the tb log.
            if USE_TB:
                tb_step += 1

            # Optimize weights and parameters using STG
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

                # Store paths
                if training_started:
                    last_training_paths.appendleft(current_path)

                    # Get current model performance for tb
                    if USE_TB:
                        training_diagnostics = evaluate_training_rollouts(
                            last_training_paths
                        )

                # Log tb variables
                if USE_TB:
                    if i % TB_FREQ == 0:

                        # Update and log learning rate tb vars
                        policy.sess.run(policy.step.assign(tb_step))
                        policy.sess.run(tb_lr_a.assign(lr_a_now))
                        policy.sess.run(tb_lr_l.assign(lr_l_now))
                        policy.sess.run(tb_lr_lag.assign(lr_a))
                        policy.tb_writer.add_summary(
                            policy.sess.run(main_sum), policy.sess.run(policy.step)
                        )

                        # Update and log other training vars to tensorboard
                        if training_started:

                            # Update and log training vars
                            policy.sess.run(
                                tb_ret.assign(training_diagnostics["return"])
                            )
                            policy.sess.run(
                                tb_len.assign(training_diagnostics["length"])
                            )
                            policy.sess.run(
                                tb_a_loss.assign(training_diagnostics["a_loss"])
                            )
                            policy.sess.run(
                                tb_lyapunov_error.assign(
                                    training_diagnostics["lyapunov_error"]
                                )
                            )
                            policy.sess.run(
                                tb_entropy.assign(training_diagnostics["entropy"])
                            )
                            policy.tb_writer.add_summary(
                                policy.sess.run(other_sum), policy.sess.run(policy.step)
                            )

                            # Log network weights
                            if WRITE_W_B:
                                policy.tb_writer.add_summary(
                                    policy.sess.run(policy.w_b_sum),
                                    policy.sess.run(policy.step),
                                )
                        policy.tb_writer.flush()

                # Decay learning rates
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                break

    # Save model and print Running time
    policy.save_result(log_dir)
    # policy.tb_writer.close()
    print("Running time: ", time.time() - t1)
    return
