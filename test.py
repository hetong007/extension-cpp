import torch
import torch.nn as nn
import numpy as np
import time

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = np.random.uniform(size=(16, 1024, 3))
x = torch.tensor(x).to(dev).type(torch.float)

B, N, C = x.shape
x_flat = x.view(-1, 3)
batch = torch.linspace(0, B-1, steps=B).repeat(N, 1).permute(1, 0).reshape(-1).to(dev).type(torch.long)

# CUDA
import cuda
fps_cuda = cuda.fps.FPS(0.2, True)

# CPU
import cpu
fps_cpu = cpu.fps.FPS(0.2, True)

# Python NATIVE
import native
fps_native = native.fps.FPS(0.2, True)

tic = time.time()
for i in range(100):
    tmp = fps_cuda(x, batch)
print(time.time() - tic)

tic = time.time()
for i in range(100):
    tmp = fps_cpu(x, batch)
print(time.time() - tic)

tic = time.time()
for i in range(100):
    tmp = fps_native(x)
print(time.time() - tic)
