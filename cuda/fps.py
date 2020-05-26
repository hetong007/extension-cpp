import math
from torch import nn
from torch.autograd import Function
import torch

import fps_cuda

torch.manual_seed(42)

class FPS(nn.Module):
    def __init__(self, npoints, ratio, random_first):
        super(FPS, self).__init__()
        self.ratio = ratio
        self.random_first = random_first

    def forward(self, pos, batch):
        return fps_cuda.fps_cuda(pos, batch, self.ratio, self.random_first)
