import torch

from ..network import BaseNetwork, create_linear_network


class ContinuousLinearQNetwork(BaseNetwork):
    def __init__(self, input_dim, output_dim, hidden_units=[],
                 initializer='xavier'):
        super(ContinuousLinearQNetwork, self).__init__()

        self.Q = create_linear_network(
            input_dim + output_dim, 1, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        Q = self.Q(x)
        return Q


class TwinnedContinuousLinearQNetwork(BaseNetwork):
    def __init__(self, input_dim, output_dim, hidden_units=[],
                 initializer='xavier'):
        super(TwinnedContinuousLinearQNetwork, self).__init__()

        self.Q1 = ContinuousLinearQNetwork(
            input_dim, output_dim, hidden_units, initializer)
        self.Q2 = ContinuousLinearQNetwork(
            input_dim, output_dim, hidden_units, initializer)

    def forward(self, states, actions):
        Q1 = self.Q1(states, actions)
        Q2 = self.Q2(states, actions)
        return Q1, Q2
