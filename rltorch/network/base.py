import torch
import torch.nn as nn


def grad_false(m):
    for param in m.parameters():
        param.requires_grad = False


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
