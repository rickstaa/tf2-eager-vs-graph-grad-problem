"""Contains the pytorch lyapunov critic. I first tried to create this as a
sequential model using the
`torch.nn.Sequential class <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_
but the network unfortunately is to difficult (Uses Square in the output).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLPLFunction(nn.Module):  # TODO: Confusing names
    """Soft Lyapunov critic Network.

    Attributes:
        q (torch.nn.modules.container.Sequential): The layers of the network.
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, use_fixed_seed=False):
        """Constructs all the necessary attributes for the Soft Q critic object.

        Args:
            obs_dim (int): Dimension of the observation space.
            act_dim (int): Dimension of the action space.
            hidden_sizes (list): Sizes of the hidden layers.
            activation (torch.nn.modules.activation): The activation function.
        """
        super().__init__()

        # Get hidden layer size
        n1 = hidden_sizes['critic'][0]

        # Setup input layer weights and biases
        if use_fixed_seed:
            torch.manual_seed(5) # FIXME: Remove random seed
        self.w1_s = nn.Parameter(torch.randn((obs_dim, n1), requires_grad=True))
        self.w1_a = nn.Parameter(torch.randn((act_dim, n1), requires_grad=True))
        self.b1 = nn.Parameter(torch.randn((1, n1), requires_grad=True))

        # Create hidden layers of the Lyapunov network
        # DEBUG: This should be similar right weighted sum as in a linear layer
        layers = []
        for i in range(1, len(hidden_sizes['critic'])):
            n = hidden_sizes['critic'][i]
            layers += [nn.Linear(hidden_sizes['critic'][i], n), nn.ReLU()]
        self.l_net = nn.Sequential(*layers)

        # FIXME: Remove random seed
        if use_fixed_seed:
            torch.manual_seed(10)
            with torch.no_grad():
                for i in range(0, len(self.l_net)-1):
                    self.l_net[i].weight = nn.Parameter(torch.randn(self.l_net[i].weight.shape, requires_grad=True))
                    self.l_net[i].bias = nn.Parameter(torch.randn(self.l_net[i].bias.shape, requires_grad=True))

    def forward(self, obs, act):
        """Perform forward pass through the network.

        Args:
            obs (torch.Tensor): The tensor of observations.

            act (torch.Tensor): The tensor of actions.

        Returns:
            torch.Tensor: The tensor containing the lyapunov values of the input
                observations and actions.
        """
        # TODO: Check if dimentions are okay?
        # FIXME: Why so low level. This can be done using linear layers?
        input_out = F.relu(torch.matmul(obs, self.w1_s) + torch.matmul(act, self.w1_a) + self.b1)
        hidden_out = self.l_net(input_out)
        l_out = torch.square(hidden_out)
        l_out = torch.sum(l_out, dim=1)
        return l_out.unsqueeze(dim=1) # Debug This looks strange why not squeeze?
