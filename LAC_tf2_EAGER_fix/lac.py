"""Minimal working version of the LAC algorithm script.
"""

import time
from collections import deque
import os
import sys
import random

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from gaussian_actor import SquashedGaussianActor
from lyapunov_critic import LyapunovCritic
from utils import evaluate_training_rollouts, get_env_from_name, training_evaluation
import logger
from pool import Pool

###############################################
# Script settings #############################
###############################################
from variant import (
    LOG_PATH,
    USE_GPU,
    ENV_NAME,
    RANDOM_SEED,
    ENV_SEED,
    TRAIN_PARAMS,
    ALG_PARAMS,
    ENV_PARAMS,
    LOG_SIGMA_MIN_MAX,
    SCALE_lambda_MIN_MAX,
    DEBUG_PARAMS,
)

# Set random seed to get comparable results for each run
if RANDOM_SEED is not None:
    os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
    os.environ["TF_CUDNN_DETERMINISTIC"] = "1"  # new flag present in tf 2.0+
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    TFP_SEED_STREAM = tfp.util.SeedStream(RANDOM_SEED, salt="tfp_1")


###############################################
# Debug options ###############################
###############################################

# Check if eager mode is enabled
print("Tensorflow eager mode enabled: " + str(tf.executing_eagerly()))

# Disable GPU if requested
if not USE_GPU:
    tf.config.set_visible_devices([], "GPU")

# Disable tf.function graph execution if debug
if DEBUG_PARAMS["debug"] and not (
    DEBUG_PARAMS["trace_net"] or DEBUG_PARAMS["trace_learn"]
):
    print(
        "WARNING: tf.functions are executed in eager mode because DEBUG=True. "
        "This significantly slow down the training. Please disable DEBUG during "
        "deployment."
    )  # TODO: MAke warning
    tf.config.experimental_run_functions_eagerly(True)

# Check for numeric errors
# tf.debugging.enable_check_numerics()

# # Enable tf2 tb debugger
# NOTE: I did not yet get this to work in eager mode
# tf.debugging.experimental.enable_dump_debug_info(
#     LOG_PATH, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1
# )


###############################################
# LAC algorithm class #########################
###############################################
# class LAC(tf.Module):  #TODO: Add so we can get a summary
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
        self.polyak = 1 - ALG_PARAMS["tau"]

        # Create network seeds
        self.ga_seeds = [
            RANDOM_SEED,
            TFP_SEED_STREAM(),
        ]  # [weight init seed, sample seed]
        self.ga_target_seeds = [
            RANDOM_SEED + 1,
            TFP_SEED_STREAM(),
        ]  # [weight init seed, sample seed]
        # self.lya_ga_target_seeds = [
        #     RANDOM_SEED,
        #     TFP_SEED_STREAM(),
        # ]  # [weight init seed, sample seed]
        self.lc_seed = RANDOM_SEED + 2  # Weight init seed
        self.lc_target_seed = RANDOM_SEED + 3  # Weight init seed

        # Determine target entropy
        if ALG_PARAMS["target_entropy"] is None:
            self.target_entropy = -self.a_dim  # lower bound of the policy entropy
        else:
            self.target_entropy = ALG_PARAMS["target_entropy"]

        # Create Learning rate placeholders
        self.LR_A = tf.Variable(ALG_PARAMS["lr_a"], name="LR_A")
        self.LR_lag = tf.Variable(ALG_PARAMS["lr_a"], name="LR_lag")
        self.LR_L = tf.Variable(ALG_PARAMS["lr_l"], name="LR_L")

        # Create lagrance multiplier placeholders
        self.log_labda = tf.Variable(tf.math.log(ALG_PARAMS["labda"]), name="lambda")
        self.log_alpha = tf.Variable(tf.math.log(ALG_PARAMS["alpha"]), name="alpha")

        ###########################################
        # Create Networks #########################
        ###########################################

        # Create Gaussian Actor (GA) and Lyapunov critic (LC) Networks
        self.ga = self._build_a(seeds=self.ga_seeds)
        self.lc = self._build_l(seed=self.lc_seed)

        # Create GA and LC target networks
        # Don't get optimized but get updated according to the EMA of the main
        # networks
        self.ga_ = self._build_a(seeds=self.ga_target_seeds)
        self.lc_ = self._build_l(seed=self.lc_target_seed)
        self.target_init()

        # Create summary writer
        if DEBUG_PARAMS["use_tb"]:
            self.step = 0
            self.tb_writer = tf.summary.create_file_writer(log_dir)

        ###########################################
        # Create optimizers #######################
        ###########################################

        self.alpha_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.lambda_train = tf.keras.optimizers.Adam(learning_rate=self.LR_lag)
        self.a_train = tf.keras.optimizers.Adam(learning_rate=self.LR_A)
        self.l_train = tf.keras.optimizers.Adam(learning_rate=self.LR_L)

        # Create model save dict
        self._save_dict = {"gaussian_actor": self.ga, "lyapunov_critic": self.lc}

        ###########################################
        # Trace networks (DEBUGGING) ##############
        ###########################################
        if DEBUG_PARAMS["use_tb"]:
            if DEBUG_PARAMS["debug"]:
                if DEBUG_PARAMS["trace_net"]:

                    # Create trace function
                    @tf.function
                    def actor_critic_trace(obs):
                        a, _, _ = self.ga(obs)
                        l = self.lc_([obs, a])
                        return l

                    # Create dummy input
                    obs = tf.random.uniform((ALG_PARAMS["batch_size"], self.s_dim))

                    # Trace networks and log to tensorboard (used for debugging)
                    tf.summary.trace_on(graph=True, profiler=True)
                    l = actor_critic_trace(obs)

                    # Write trace to tensorboard
                    with self.tb_writer.as_default():
                        tf.summary.trace_export(
                            name="actor_critic_trace", step=0, profiler_outdir=log_dir
                        )

    @tf.function
    def choose_action(self, s, evaluation=False):
        """Returns the current action of the policy.

        Args:
            s (np.numpy): The current state.
            evaluation (bool, optional): Whether to return a deterministic action.
            Defaults to False.

        Returns:
            np.numpy: The current action.
        """

        # Make sure s is float32 tensorflow tensor
        if not isinstance(s, tf.Tensor):
            s = tf.convert_to_tensor(s, dtype=tf.float32)
        elif s.dtype != tf.float32:
            s = tf.cast(s, dtype=tf.float32)

        # Get current best action
        if evaluation is True:
            try:
                _, deterministic_a, _ = self.ga(tf.reshape(s, (1, -1)))
                return deterministic_a[0]
            except ValueError:
                return
        else:
            a, _, _ = self.ga(tf.reshape(s, (1, -1)))
            return a[0]

    @tf.function
    def learn(self, LR_A, LR_L, LR_lag, batch):
        """Runs the SGD to update all the optimize parameters.

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

        # Update target networks
        self.update_target()

        # Get Lyapunov target
        a_, _, _ = self.ga_(bs_)
        l_ = self.lc_([bs_, a_])
        l_target = br + ALG_PARAMS["gamma"] * (1 - bterminal) * tf.stop_gradient(l_)

        # Lyapunov candidate constraint function graph
        with tf.GradientTape() as l_tape:

            # Calculate current lyapunov value
            l = self.lc([bs, ba])

            # Calculate L_backup
            l_error = tf.compat.v1.losses.mean_squared_error(
                labels=l_target, predictions=l
            )

        # Actor loss and optimizer graph
        with tf.GradientTape() as a_tape:

            # Calculate current value and target lyapunov multiplier value
            lya_a_, _, _ = self.ga(bs_)
            lya_l_ = self.lc([bs_, lya_a_])

            # Calculate Lyapunov constraint function
            self.l_delta = tf.reduce_mean(lya_l_ - l + (ALG_PARAMS["alpha3"]) * br)

            # Calculate log probability of a_input based on current policy
            _, _, log_pis = self.ga(bs)

            # Calculate actor loss
            a_loss = self.labda * self.l_delta + self.alpha * tf.reduce_mean(log_pis)

        # Lagrance multiplier loss functions and optimizers graphs
        with tf.GradientTape() as lambda_tape:
            labda_loss = -tf.reduce_mean(self.log_labda * self.l_delta)

        # Calculate alpha loss
        with tf.GradientTape() as alpha_tape:
            alpha_loss = -tf.reduce_mean(
                self.log_alpha * tf.stop_gradient(log_pis + self.target_entropy)
            )  # Trim down

        # Apply lambda gradients
        lambda_grads = lambda_tape.gradient(labda_loss, [self.log_labda])
        self.lambda_train.apply_gradients(zip(lambda_grads, [self.log_labda]))

        # Apply alpha gradients
        alpha_grads = alpha_tape.gradient(alpha_loss, [self.log_alpha])
        self.alpha_train.apply_gradients(zip(alpha_grads, [self.log_alpha]))

        # Apply actor gradients
        a_grads = a_tape.gradient(a_loss, self.ga.trainable_variables)
        self.a_train.apply_gradients(zip(a_grads, self.ga.trainable_variables))

        # Apply critic gradients
        l_grads = l_tape.gradient(l_error, self.lc.trainable_variables)
        self.l_train.apply_gradients(zip(l_grads, self.lc.trainable_variables))

        # Return results
        return (
            self.labda,
            self.alpha,
            l_error,
            tf.reduce_mean(tf.stop_gradient(-log_pis)),
            a_loss,
        )

    def _build_a(self, name="gaussian_actor", trainable=True, seeds=[None, None]):
        """Setup SquashedGaussianActor Graph.

        Args:
            name (str, optional): Network name. Defaults to "gaussian_actor".

            trainable (bool, optional): Whether the weights of the network layers should
                be trainable. Defaults to True.

            seeds (list, optional): The random seeds used for the weight initialization
                and the sampling ([weights_seed, sampling_seed]). Defaults to
                [None, None]
        Returns:
            tuple: Tuple with network output tensors.
        """

        # Return GA
        return SquashedGaussianActor(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["actor"],
            name=name,
            log_std_min=LOG_SIGMA_MIN_MAX[0],
            log_std_max=LOG_SIGMA_MIN_MAX[1],
            trainable=trainable,
            seeds=seeds,
        )

    def _build_l(self, name="lyapunov_critic", trainable=True, seed=None):
        """Setup lyapunov critic graph.

        Args:
            name (str, optional): Network name. Defaults to "lyapunov_critic".

            trainable (bool, optional): Whether the weights of the network layers should
                be trainable. Defaults to True.

            seed (int, optional): The seed used for the weight initialization. Defaults
                to None.

        Returns:
            tuple: Tuple with network output tensors.
        """

        # Return GA
        # TODO: Check if trainable is needed
        return LyapunovCritic(
            obs_dim=self.s_dim,
            act_dim=self.a_dim,
            hidden_sizes=self.network_structure["critic"],
            name=name,
            trainable=trainable,
            seed=seed,
        )

    def save_result(self, path):
        """Save current policy.

        Args:
            path (str): The path where you want to save the policy.
        """

        # Save all models/tensors in the _save_dict
        for name, model in self._save_dict.items():
            save_path = path + "/policy/" + name
            model.save_weights(save_path)
            print(f"Saved '{name}' weights to path: {save_path}")

    def restore(self, path):
        """Restore policy.

        Args:
            path (str): The path where you want to save the policy.

        Returns:
            bool: Boolean specifying whether the policy was loaded successfully.
        """

        # Check if the models exist
        checkpoints = [
            f.replace(".index", "") for f in os.listdir(path) if f.endswith(".index")
        ]
        if not checkpoints:
            success_load = False
            return success_load

        # Load the model weights
        self.ga.load_weights(path + "/gaussian_actor")
        self.lc.load_weights(path + "/lyapunov_critic")
        self.target_init()
        success_load = True
        return success_load

    @property
    def alpha(self):
        return tf.exp(self.log_alpha)

    @property
    def labda(self):
        return tf.clip_by_value(tf.exp(self.log_labda), *SCALE_lambda_MIN_MAX)

    @tf.function
    def target_init(self):
        # Initializing targets to match main variables
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(pi_main)
        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(l_main)

    @tf.function
    def update_target(self):
        # Polyak averaging for target variables
        # (control flow because sess.run otherwise evaluates in nondeterministic order)
        for pi_main, pi_targ in zip(self.ga.variables, self.ga_.variables):
            pi_targ.assign(self.polyak * pi_targ + (1 - self.polyak) * pi_main)

        for l_main, l_targ in zip(self.lc.variables, self.lc_.variables):
            l_targ.assign(self.polyak * l_targ + (1 - self.polyak) * l_main)


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
    last_training_paths = deque(maxlen=TRAIN_PARAMS["num_of_training_paths"])
    training_started = False

    # Log initial values to tensorboard
    if DEBUG_PARAMS["use_tb"]:

        # Trace learn method (Used for debugging)
        if DEBUG_PARAMS["debug"]:
            if DEBUG_PARAMS["trace_net"]:

                # Create dummy input
                batch = {
                    "s": tf.random.uniform((ALG_PARAMS["batch_size"], policy.s_dim)),
                    "a": tf.random.uniform((ALG_PARAMS["batch_size"], policy.a_dim)),
                    "r": tf.random.uniform((ALG_PARAMS["batch_size"], 1)),
                    "terminal": tf.zeros((ALG_PARAMS["batch_size"], 1)),
                    "s_": tf.random.uniform((ALG_PARAMS["batch_size"], policy.s_dim)),
                }

                # Trace learn method and log to tensorboard
                tf.summary.trace_on(graph=True, profiler=True)
                policy.learn(lr_a_now, lr_l_now, lr_a, batch)
                with policy.tb_writer.as_default():
                    tf.summary.trace_export(
                        name="learn", step=0, profiler_outdir=log_dir
                    )

            # Shut down as we are in debug mode
            if DEBUG_PARAMS["trace_net"] or DEBUG_PARAMS["trace_learn"]:
                print(
                    "Shutting down training as a trace was requested in debug mode. "
                    "This was done since during the trace a backward pass was performed "
                    "on dummy data. Please disable the trace to continue training "
                    "while being in debug mode."
                )
                sys.exit(0)

        # Log initial values
        with policy.tb_writer.as_default():
            tf.summary.scalar("lr_a", lr_a_now, step=0)
            tf.summary.scalar("lr_l", lr_l_now, step=0)
            tf.summary.scalar("lr_lag", lr_a, step=0)
            tf.summary.scalar("alpha", policy.alpha, step=0)
            tf.summary.scalar("lambda", policy.labda, step=0)

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

            # Increment tensorboard step counter
            # NOTE: This was done differently from the global_step counter since
            # otherwise there were inconsistencies in the tb log.
            if DEBUG_PARAMS["use_tb"]:
                policy.step += 1

            # Store experience in replay buffer
            pool.store(s, a, r, terminal, s_)

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
                    if DEBUG_PARAMS["use_tb"]:
                        training_diagnostics = evaluate_training_rollouts(
                            last_training_paths
                        )

                # Log tb variables
                if DEBUG_PARAMS["use_tb"]:
                    if i % DEBUG_PARAMS["tb_freq"] == 0:

                        # Log learning rate to tb
                        with policy.tb_writer.as_default():
                            tf.summary.scalar("lr_a", lr_a_now, step=policy.step)
                            tf.summary.scalar("lr_l", lr_l_now, step=policy.step)
                            tf.summary.scalar("lr_lag", lr_a, step=policy.step)
                            tf.summary.scalar("alpha", policy.alpha, step=policy.step)
                            tf.summary.scalar("lambda", policy.labda, step=policy.step)

                        # Update and log other training vars to tensorboard
                        if training_started:
                            with policy.tb_writer.as_default():
                                tf.summary.scalar(
                                    "ep_ret",
                                    training_diagnostics["return"],
                                    step=policy.step,
                                )
                                tf.summary.scalar(
                                    "ep_length",
                                    training_diagnostics["length"],
                                    step=policy.step,
                                )
                                tf.summary.scalar(
                                    "a_loss",
                                    training_diagnostics["a_loss"],
                                    step=policy.step,
                                )
                                tf.summary.scalar(
                                    "lyapunov_error",
                                    training_diagnostics["lyapunov_error"],
                                    step=policy.step,
                                )
                                tf.summary.scalar(
                                    "entropy",
                                    training_diagnostics["entropy"],
                                    step=policy.step,
                                )

                            # Log network weights
                            if DEBUG_PARAMS["write_w_b"]:
                                with policy.tb_writer.as_default():

                                    # GaussianActor weights/biases
                                    tf.summary.histogram(
                                        "Ga/l1/weights",
                                        policy.ga.net_0.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/l1/bias",
                                        policy.ga.net_0.weights[1],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/l2/weights",
                                        policy.ga.net_1.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/l2/bias",
                                        policy.ga.net_1.weights[1],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/mu/weights",
                                        policy.ga.mu.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/mu/bias",
                                        policy.ga.mu.weights[1],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/log_sigma/weights",
                                        policy.ga.log_sigma.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga/log_sigma/bias",
                                        policy.ga.log_sigma.weights[1],
                                        step=policy.step,
                                    )

                                    # Target GaussianActor weights/biases
                                    tf.summary.histogram(
                                        "Ga_/l1/weights",
                                        policy.ga_.net_0.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/l1/bias",
                                        policy.ga_.net_0.weights[1],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/l2/weights",
                                        policy.ga_.net_1.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/l2/bias",
                                        policy.ga_.net_1.weights[1],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/mu/weights",
                                        policy.ga_.mu.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/mu/bias",
                                        policy.ga_.mu.weights[1],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/log_sigma/weights",
                                        policy.ga_.log_sigma.weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Ga_/log_sigma/bias",
                                        policy.ga_.log_sigma.weights[1],
                                        step=policy.step,
                                    )

                                    # Lyapunov critic weights/biases
                                    tf.summary.histogram(
                                        "Lc/w1_s", policy.lc.w1_s, step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc/w1_a", policy.lc.w1_a, step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc/b1", policy.lc.b1, step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc/net/l2/weights",
                                        policy.lc.net.layers[0].weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc/net/l2/bias",
                                        policy.lc.net.layers[0].weights[1],
                                        step=policy.step,
                                    )

                                    # Target Lyapunov critic weights/biases
                                    tf.summary.histogram(
                                        "Lc_/w1_s", policy.lc_.w1_s, step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc_/w1_a", policy.lc_.w1_a, step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc_/b1", policy.lc_.b1, step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc_/net/l2/weights",
                                        policy.lc_.net.layers[0].weights[0],
                                        step=policy.step,
                                    )
                                    tf.summary.histogram(
                                        "Lc_/net/l2/bias",
                                        policy.lc_.net.layers[0].weights[1],
                                        step=policy.step,
                                    )

                # Decay learning rates
                frac = 1.0 - (global_step - 1.0) / ENV_PARAMS["max_global_steps"]
                lr_a_now = lr_a * frac  # learning rate for actor, lambda, alpha
                lr_l_now = lr_l * frac  # learning rate for lyapunov critic
                break

    # Save model and print Running time
    policy.save_result(log_dir)
    print("Running time: ", time.time() - t1)
    return
