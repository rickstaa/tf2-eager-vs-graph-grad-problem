"""Contains the pytorch lyapunov critic. I first tried to create this as a
sequential model using the
`torch.nn.Sequential class <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_
but the network unfortunately is to difficult (Uses Square in the output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def mlp(sizes, activation, output_activation=nn.Identity):
    """Create a multi-layered perceptron using pytorch.

    Args:
        sizes (list): The size of each of the layers.

        activation (torch.nn.modules.activation): The activation function used for the
            hidden layers.

        output_activation (torch.nn.modules.activation, optional): The activation
            function used for the output layers. Defaults to torch.nn.Identity.

    Returns:
        torch.nn.modules.container.Sequential: The multi-layered perceptron.
    """
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)

class MLPLFunction(nn.Module):  # TODO: Confusing names
    """Soft Lyapunov critic Network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation=nn.ReLU, use_fixed_seed=False):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function.
        """
        super().__init__()
        self.l = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation) # Check if neede
        # self.l = mlp([obs_dim + act_dim] + list(hidden_sizes), activation)

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the lyapunov values of the input
                observations and actions.
        """
        l_hid_out = self.l(torch.cat([obs, act], dim=-1))
        l_out = torch.square(l_hid_out)
        l_out = torch.sum(l_out, dim=1)
        return l_out.unsqueeze(dim=1) # FIXME: Critical to ensure q has right shape.
        # return torch.squeeze(l_out, -1)  # FIXME: Critical to ensure q has right shape.
