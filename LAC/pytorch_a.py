"""Contains the pytorch actor. I first tried to create this as a
sequential model using the
`torch.nn.Sequential class <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_
but the network unfortunately is to difficult since it has multiple outputs.
"""
# NOTE: THIS SQUASHED GAUSSIAN ACTOR IS NOT CLAMPED!

import numpy as np
import torch
import torch.nn as nn

import torch.nn.functional as F
from torch.distributions.normal import Normal

class SquashedGaussianMLPActor(nn.Module):
    """The squashed gaussian actor network.

    Attributes:
        net (torch.nn.modules.container.Sequential): The input/hidden layers of the
            network.

        mu (torch.nn.modules.linear.Linear): The output layer which returns the mean of
            the actions.

        log_std_layer (torch.nn.modules.linear.Linear): The output layer which returns
            the log standard deviation of the actions.

        act_limit (np.float32): Scaling factor used for the actions that come out of
            the network.
    """

    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_sizes,
        log_std_min=-20,
        log_std_max=2.0,
        use_fixed_seed=False,
    ):
        """Constructs all the necessary attributes for the Squashed Gaussian Actor
        object.

        Args:
            obs_dim (int): Dimension of the observation space.

            act_dim (int): Dimension of the action space.

            hidden_sizes (list): Sizes of the hidden layers.

            log_std_min (int, optional): The minimum log standard deviation. Defaults
                to -20.

            log_std_max (float, optional): The maximum log standard deviation. Defaults
                to 2.0.
        """
        super().__init__()

        # Get class parameters
        self._log_std_min = log_std_min
        self._log_std_max = log_std_max

        # Get hidden layer structure
        n1 = hidden_sizes['actor'][0]  # Size of hidden layer one
        n2 = hidden_sizes['actor'][1]  # Size of hidden layer tw

        # Create nn layers
        net_0 = nn.Sequential(nn.Linear(obs_dim, n1), nn.ReLU())
        net_1 = nn.Sequential(nn.Linear(n1, n2), nn.ReLU())
        self.net = nn.Sequential(net_0, net_1)
        self.mu_layer = nn.Linear(n2, act_dim)
        self.log_sigma = nn.Linear(n2, act_dim)

        # FIXME: Remove after testing
        if use_fixed_seed:
            torch.manual_seed(0)
            with torch.no_grad():
                self.net[0][0].weight = nn.Parameter(torch.randn(self.net[0][0].weight.shape, requires_grad=True))
                self.net[0][0].bias = nn.Parameter(torch.randn(self.net[0][0].bias.shape, requires_grad=True))
                self.net[1][0].weight = nn.Parameter(torch.randn(self.net[1][0].weight.shape, requires_grad=True))
                self.net[1][0].bias = nn.Parameter(torch.randn(self.net[1][0].bias.shape, requires_grad=True))
                self.mu_layer.weight = nn.Parameter(torch.randn(self.mu_layer.weight.shape, requires_grad=True))
                self.mu_layer.bias = nn.Parameter(torch.randn(self.mu_layer.bias.shape, requires_grad=True))
                self.log_sigma.weight = nn.Parameter(torch.randn(self.log_sigma.weight.shape, requires_grad=True))
                self.log_sigma.bias = nn.Parameter(torch.randn(self.log_sigma.bias.shape, requires_grad=True))

    def forward(self, obs, deterministic=False, with_logprob=True):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            deterministic (bool, optional): Whether we want to use a deterministic
                policy (used at test time). When true the mean action of the stochastic
                policy is returned. If false the action is sampled from the stochastic
                policy. Defaults to False.

            with_logprob (bool, optional): Whether we want to return the log probability
                of an action. Defaults to True.

        Returns:
            torch.Tensor,  torch.Tensor: The actions given by the policy, the log
            probabilities of each of these actions.
        """

        # Calculate required variables
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_sigma = self.log_sigma(net_out)
        log_sigma = torch.clamp(log_sigma, self._log_std_min, self._log_std_max)
        sigma = torch.exp(log_sigma)

        # Check summing axis
        sum_axis = 0 if obs.shape.__len__() == 1 else 1

        # Pre-squash distribution and sample
        # TODO: Check if this has the right size. LAC samples from base distribution
        # Sample size is memory buffer size!
        pi_distribution = Normal(mu, sigma)
        raw_action = (
            pi_distribution.rsample()
            # DEBUG: The tensorflow implmentation samples
        )  # Sample while using the parameterization trick

        # Compute log probability in squashed gaussian
        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh
            # squashing. NOTE: The correction formula is a little bit magic. To get an
            # understanding of where it comes from, check out the original SAC paper
            # (arXiv 1801.01290) and look in appendix C. This is a more
            # numerically-stable equivalent to Eq 21. Try deriving it yourself as a
            # (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(raw_action).sum(axis=-1)
            logp_pi -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(
                axis=1
            )
        else:
            logp_pi = None

        # Calculate scaled action and return the action and its log probability
        clipped_a = torch.tanh(raw_action)  # Squash gaussian to be between -1 and 1

        # Get clipped mu
        # FIXME: Is this okay LAC also squashes this output?!
        # clipped_mu = torch.tanh(mu) # LAC version
        clipped_mu = mu

        # Return action and log likelihood
        # Debug: The LAC expects a distribution we already return the log probabilities
        return clipped_a, clipped_mu, logp_pi