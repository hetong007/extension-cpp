import math, os
from torch import nn
from torch.autograd import Function
from torch.utils.cpp_extension import load
import torch

import fps_cuda

torch.manual_seed(42)

class FPS(nn.Module):
    def __init__(self, ratio, random_first):
        super(FPS, self).__init__()
        self.ratio = ratio
        self.random_first = random_first

    def forward(self, pos, batch=None):
        if batch is not None:
            assert pos.size(0) == batch.numel()
            batch_size = int(batch.max()) + 1

            deg = pos.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch, torch.ones_like(batch))

            ptr = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr[1:])
        else:
            ptr = torch.tensor([0, pos.size(0)], device=pos.device)

        return fps_cuda.fps(pos, ptr, self.ratio, self.random_first)

class FPS_jit(nn.Module):
    def __init__(self, ratio, random_first, path='.'):
        super(FPS_jit, self).__init__()
        self.ratio = ratio
        self.random_first = random_first
        self.func = load('fps_cuda', [os.path.join(path, 'fps_cuda.cpp'),
                                      os.path.join(path, 'fps_cuda_kernel.cu')], verbose=False)

    def forward(self, pos, batch=None):
        if batch is not None:
            assert pos.size(0) == batch.numel()
            batch_size = int(batch.max()) + 1

            deg = pos.new_zeros(batch_size, dtype=torch.long)
            deg.scatter_add_(0, batch, torch.ones_like(batch))

            ptr = deg.new_zeros(batch_size + 1)
            torch.cumsum(deg, 0, out=ptr[1:])
        else:
            ptr = torch.tensor([0, pos.size(0)], device=pos.device)

        return self.func.fps(pos, ptr, self.ratio, self.random_first)
