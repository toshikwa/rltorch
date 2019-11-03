import torch
from torch.distributions import Normal

from rltorch.network import BaseNetwork, create_linear_network


class LinearGaussianPolicy(BaseNetwork):
    LOG_STD_MAX = 2
    LOG_STD_MIN = -20
    eps = 1e-6

    def __init__(self, input_dim, output_dim, hidden_units=[],
                 initializer='xavier'):
        super(LinearGaussianPolicy, self).__init__()

        self.policy = create_linear_network(
            input_dim, output_dim*2, hidden_units=hidden_units,
            initializer=initializer)

    def forward(self, states):
        mean, log_std = torch.chunk(self.policy(states), 2, dim=-1)
        log_std = torch.clamp(
            log_std, min=self.LOG_STD_MIN, max=self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, states):
        mean, log_std = self.forward(states)
        std = log_std.exp()

        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)

        log_prob = normal.log_prob(x_t)\
            - torch.log(1 - action.pow(2) + self.eps)
        entropy = -log_prob.sum(1, keepdim=True)

        return action, entropy, torch.tanh(mean)
