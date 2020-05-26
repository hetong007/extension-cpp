import math
from torch import nn
from torch.autograd import Function
import torch

import fps_cpu

torch.manual_seed(42)

class FPS(nn.Module):
    def __init__(self, npoints, batch, ratio, random_first):
        super(LLTM, self).__init__()
        self.npoints = npoints
        self.batch = batch
        self.ratio = ratio
        self.random_first = random_first

    def forward(self, pos):
        return fps_cpu(pos, batch, ratio, random_first)
