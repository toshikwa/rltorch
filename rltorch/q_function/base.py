import torch
import torch.nn as nn


def weights_init_he(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def weights_init_xavier(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def grad_false(m):
    for param in m.parameters():
        param.requires_grad = False


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class BaseNetwork(nn.Module):

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def eval(self):
        return super(BaseNetwork, self).eval().apply(grad_false)

    @property
    def n_params(self):
        n = 0
        for p in list(self.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            n += nn
        return n
