import torch.nn as nn

from .base import weights_init_xavier, Flatten, BaseNetwork


def create_conv_base(input_channels):
    return nn.Sequential(
        nn.Conv2d(input_channels, 32, kernel_size=8, stride=4, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
        nn.ReLU(inplace=True),
        Flatten(),
    )


class ConvQNetwork(BaseNetwork):
    def __init__(self, input_channels, num_actions):
        super(ConvQNetwork, self).__init__()

        self.base = create_conv_base(
            input_channels).apply(weights_init_xavier)

        self.V_stream = nn.Sequential(
            nn.Linear(7 * 7 * 64, 512),
            nn.ReLU(True),
            nn.Linear(512, 1)
        ).apply(weights_init_xavier)

        self.A_stream = nn.Sequential(
            nn.Linear(7*7*64, 512),
            nn.ReLU(True),
            nn.Linear(512, num_actions)
        ).apply(weights_init_xavier)

    def forward(self, states):
        h = self.base(states)
        V = self.V_stream(h)
        A = self.A_stream(h)
        Q = V + A - A.mean(1, keepdim=True)
        return Q
