import torch.nn as nn


class ReshapeLayer(nn.Module):
    def __init__(self, s1, s2=None):
        super(ReshapeLayer, self).__init__()
        if s2 is None:
            s2 = s1
        self.s1 = s1
        self.s2 = s2

    def forward(self, x):
        return x.view(-1, 1, self.s1, self.s2)
