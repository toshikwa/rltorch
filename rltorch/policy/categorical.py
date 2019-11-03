import torch
from torch import nn
from torch.distributions import Categorical

from rltorch.network import BaseNetwork, create_linear_network, create_dqn_base


class ConvCategoricalPolicy(BaseNetwork):

    def __init__(self, num_channels, output_dim, initializer='kaiming'):
        super(ConvCategoricalPolicy, self).__init__()

        self.policy = nn.Sequential(
            *create_dqn_base(num_channels),
            *create_linear_network(
                7*7*64, output_dim, hidden_units=[512],
                output_activation='softmax',
                initializer=initializer)
            )

    def forward(self, states):
        action_probs = self.policy(states)
        return action_probs

    def sample(self, state):
        action_probs = self.policy(state)
        greedy_actions = torch.argmax(action_probs, dim=1, keepdim=True)

        categorical = Categorical(action_probs)
        actions = categorical.sample().view(-1, 1)

        log_action_probs = torch.log(
            action_probs + (action_probs == 0.0).float() * 1e-8)

        return actions, action_probs, log_action_probs, greedy_actions
