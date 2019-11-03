from ..network import BaseNetwork, create_dqn_base, create_linear_network


class DiscreteConvQNetwork(BaseNetwork):
    def __init__(self, num_channels, output_dim):
        super(DiscreteConvQNetwork, self).__init__()

        self.base = create_dqn_base(num_channels)
        self.V_stream = create_linear_network(
            7*7*64, 1, hidden_units=[512])
        self.A_stream = create_linear_network(
            7*7*64, output_dim, hidden_units=[512])

    def forward(self, states):
        h = self.base(states)
        V = self.V_stream(h)
        A = self.A_stream(h)
        Q = V + A - A.mean(1, keepdim=True)
        return Q
