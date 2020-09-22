"""Validate if the Pytorch and Tensorflow networks are the same."""
import torch
from torchviz import make_dot, make_dot_from_trace
import datetime

import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import tensorboard

from pytorch_a import SquashedGaussianMLPActor
from pytorch_l import MLPLFunction

from squash_bijector import SquashBijector

# FIXME! REMOVE LATER!
torch.manual_seed(0)
torch.cuda.manual_seed(0)
tf.reset_default_graph()
tf.set_random_seed(0)
random.seed(0)
np.random.seed(0)

# Script parameters
SCALE_DIAG_MIN_MAX = (-20, 2)
SCALE_lambda_MIN_MAX = (0, 1)

S_DIM = 2
A_DIM = 2

NETWORK_STRUCTURE = {"critic": [128, 128], "actor": [64, 64]}


def _build_a(s, name="actor", reuse=None, custom_getter=None):
    global S_DIM
    global A_DIM
    global NETWORK_STRUCTURE

    if USE_PYTORCH:

        # ===============================
        # BEGIN >>> Pytorch CODE ========
        # ===============================
        # Get action and observation space size

        # Create and return Squashed Gaussian actor
        SGA = SquashedGaussianMLPActor(
            S_DIM,
            A_DIM,
            NETWORK_STRUCTURE,
            log_std_min=SCALE_DIAG_MIN_MAX[0],
            log_std_max=SCALE_DIAG_MIN_MAX[1],
        )
        return SGA
        # ===============================
        # END <<<<< Pytorch CODE ========
        # ===============================
    else:
        if reuse is None:
            trainable = True
        else:
            trainable = False

        S_DIM = 2
        A_DIM = 2
        w_a = tf.zeros([S_DIM, A_DIM])
        print(w_a)

        with tf.variable_scope(name, reuse=reuse, custom_getter=custom_getter):

            batch_size = tf.shape(s)[0]
            squash_bijector = SquashBijector()
            base_distribution = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(A_DIM), scale_diag=tf.ones(A_DIM)
            )
            epsilon = base_distribution.sample(batch_size)

            ## Construct the feedforward action
            n1 = NETWORK_STRUCTURE["actor"][0]
            n2 = NETWORK_STRUCTURE["actor"][1]

            net_0 = tf.layers.dense(
                s, n1, activation=tf.nn.relu, name="l1", trainable=trainable
            )  # 原始是30
            net_1 = tf.layers.dense(
                net_0, n2, activation=tf.nn.relu, name="l4", trainable=trainable
            )  # 原始是30
            mu = tf.layers.dense(
                net_1, A_DIM, activation=None, name="a", trainable=trainable
            )
            log_sigma = tf.layers.dense(net_1, A_DIM, None, trainable=trainable)

            # log_sigma = tf.layers.dense(s, A_DIM, None, trainable=trainable)

            log_sigma = tf.clip_by_value(log_sigma, *SCALE_DIAG_MIN_MAX)
            sigma = tf.exp(log_sigma)

            bijector = tfp.bijectors.Affine(shift=mu, scale_diag=sigma)
            raw_action = bijector.forward(epsilon)
            clipped_a = squash_bijector.forward(raw_action)

            ## Construct the distribution
            bijector = tfp.bijectors.Chain(
                (squash_bijector, tfp.bijectors.Affine(shift=mu, scale_diag=sigma),)
            )
            distribution = tfp.distributions.ConditionalTransformedDistribution(
                distribution=base_distribution, bijector=bijector
            )

            clipped_mu = squash_bijector.forward(mu)

        return clipped_a, clipped_mu, distribution


def _build_l(s, a, reuse=None, custom_getter=None):
    global S_DIM
    global A_DIM
    global NETWORK_STRUCTURE

    if USE_PYTORCH:

        # ===============================
        # BEGIN >>> Pytorch CODE ========
        # ===============================

        # Create and return Squashed Gaussian actor
        LC = MLPLFunction(S_DIM, A_DIM, NETWORK_STRUCTURE)
        # l = LC(torch.cat([s.unsqueeze(0),s.unsqueeze(0)]), a) # test network
        return LC

        # # Get hidden layer size
        # n1 = NETWORK_STRUCTURE['critic'][0]

        # # Create hidden layers of the Lyapunov network
        # # FIXME: Why so low level. This can be done using linear layers?
        # # DEBUG: This should be similar right weighted sum as in a linear layer
        # layers = []
        # w1_s = torch.randn((S_DIM, n1), requires_grad=True)
        # w1_a = torch.randn((A_DIM, n1), requires_grad=True)
        # b1 = torch.randn((1, n1), requires_grad=True)
        # net_0 = F.relu(torch.matmul(s, w1_s) + torch.matmul(a, w1_a) + b1)
        # layers.append(net_0)
        # for i in range(1, len(NETWORK_STRUCTURE['critic'])):
        #     n = NETWORK_STRUCTURE['critic'][i]
        #     layers += [nn.Linear(NETWORK_STRUCTURE['critic'][i], n), nn.ReLU()]

        # # Create Output layer
        # torch.sum(torch.square(layers[-1]), dim=1)

        # # Return network
        # return nn.Sequential(*layers)

        # ===============================
        # END <<<<< Pytorch CODE ========
        # ===============================
    else:
        trainable = True if reuse is None else False

        with tf.variable_scope("Lyapunov", reuse=reuse, custom_getter=custom_getter):
            n1 = NETWORK_STRUCTURE["critic"][0]

            layers = []
            w1_s = tf.get_variable("w1_s", [S_DIM, n1], trainable=trainable)
            w1_a = tf.get_variable("w1_a", [A_DIM, n1], trainable=trainable)
            b1 = tf.get_variable("b1", [1, n1], trainable=trainable)
            net_0 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            layers.append(net_0)
            for i in range(1, len(NETWORK_STRUCTURE["critic"])):
                n = NETWORK_STRUCTURE["critic"][i]
                layers.append(
                    tf.layers.dense(
                        layers[i - 1],
                        n,
                        activation=tf.nn.relu,
                        name="l" + str(i + 1),
                        trainable=trainable,
                    )
                )

            return tf.expand_dims(
                tf.reduce_sum(tf.square(layers[-1]), axis=1), axis=1
            )  # Q(s,a)


if __name__ == "__main__":

    ###############################
    # Create pytorch networks #####
    ###############################
    USE_PYTORCH = True
    ga = _build_a(S_DIM)  # 这个网络用于及时更新参数
    lc = _build_l(S_DIM, A_DIM)  # lyapunov 网络

    # Create target networks
    ga_ = _build_a(S_DIM)
    lc_ = _build_l(S_DIM, A_DIM)

    # Freeze target networks
    for p in ga_.parameters():
        p.requires_grad = False
    for p in lc_.parameters():
        p.requires_grad = False

    # Create untrainable lyapunov actor and l_target
    lya_ga_ = _build_a(S_DIM)
    lya_lc_ = _build_l(S_DIM, A_DIM)

    # Make the lyapunov actor un-trainable
    for p in lya_ga_.parameters():
        p.requires_grad = False
    for p in lya_lc_.parameters():
        p.requires_grad = False

    # Vizualize networks
    print("===PYTORCH NETWORKS===")
    print("")
    print("----GA----")
    print(ga)
    print("")
    print("----LC----")
    print(lc)
    print("")
    print("----GA_----")
    print(ga_)
    print("")
    print("----lc_----")
    print(lc_)
    print("")
    print("----lya_ga_----")
    print(lya_ga_)
    print("")
    print("----lya_ga_----")
    print(lya_lc_)

    # Create dummy input
    S = torch.randn(1, S_DIM)
    A = torch.randn(1, S_DIM)

    # Vizualize network graph
    # print("\nVizualize GA")
    # ga_dot = make_dot(ga(S), params=dict(ga.named_parameters()))
    # ga_dot.format = 'png'
    # ga_dot.render(filename="Pytorch_ga_graph")
    # print("\nVizualize GA")
    # lc_dot = make_dot(lc(S, A), params=dict(lc.named_parameters()))
    # lc_dot.format = 'png'
    # lc_dot.render(filename="Pytorch_lc_graph")

    ###############################
    # Create tensorflow networks ##
    ###############################
    USE_PYTORCH = False
    # Print graphs to tensorboard
    with tf.Session() as sess:

        # Create summary writer
        summaryMerged = tf.summary.merge_all()
        filename = "./summary_log/run" + datetime.datetime.now().strftime(
            "%Y-%m-%d--%H-%M-%s"
        )
        writer = tf.summary.FileWriter(filename, sess.graph)

        with tf.variable_scope("Actor"):
            # Create placeholders
            S = tf.placeholder(tf.float32, [None, S_DIM], "s")
            S_ = tf.placeholder(tf.float32, [None, S_DIM], "s_")
            a_input = tf.placeholder(tf.float32, [None, A_DIM], "a_input")
            a_input_ = tf.placeholder(tf.float32, [None, A_DIM], "a_input_")

            # Create networks
            a, deterministic_a, a_dist = _build_a(S)
            l = _build_l(S, a_input)

            # Get network parameters
            a_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/actor"
            )
            l_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, scope="Actor/Lyapunov"
            )

            # Ema
            tau = 5e-3
            ema = tf.train.ExponentialMovingAverage(decay=1 - tau)  # soft replacement

            def ema_getter(getter, name, *args, **kwargs):
                return ema.average(getter(name, *args, **kwargs))

            target_update = [
                ema.apply(a_params),
                ema.apply(l_params),
            ]  # soft update operation

            # Create target networks
            a_, _, a_dist_ = _build_a(
                S_, reuse=True, custom_getter=ema_getter
            )  # replaced target parameters
            l_ = _build_l(S_, a_, reuse=True, custom_getter=ema_getter)
            log_pis = log_pis = a_dist.log_prob(a)
            prob = tf.reduce_mean(a_dist.prob(a))

            # Create Lyapunov target networks
            lya_a_, _, _ = _build_a(S_, reuse=True)
            l_ = _build_l(S_, lya_a_, reuse=True)

            # Initialize network parameters
            sess.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter("output", sess.graph)
            print(sess.run(deterministic_a, {S: s}))
            writer.close()
