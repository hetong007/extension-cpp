import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.cpp_extension import load

cuda_dev = torch.device("cuda")
cpu_dev = torch.device("cpu")

x = np.random.uniform(size=(16, 1024, 3))
x = torch.tensor(x).type(torch.float)
x_gpu = x.to(cuda_dev)
B, N, C = x.shape

x_flat_cpu = x.view(-1, 3).to(cpu_dev)
x_flat_gpu = x_flat_cpu.to(cuda_dev)
batch_cpu = torch.linspace(0, B-1, steps=B).repeat(N, 1).permute(1, 0).reshape(-1).type(torch.long).to(cpu_dev)
batch_gpu = batch_cpu.to(cuda_dev)

# CUDA
import cuda.fps as cuda_fps
fps_cuda = cuda_fps.FPS(0.2, True)

# CPU
import cpu.fps as cpu_fps
fps_cpu = cpu_fps.FPS(0.2, True)

# CUDA jit
import cuda.fps as cuda_fps
fps_cuda_jit = cuda_fps.FPS_jit(0.2, True, 'cuda')

# CPU jit
import cpu.fps as cpu_fps
fps_cpu_jit = cpu_fps.FPS_jit(0.2, True, 'cpu')

# Python NATIVE
import native.fps as native_fps
fps_native = native_fps.FPS(0.2, True)

# Timing
N = 20
tic = time.time()
for i in range(N):
    tmp = fps_cpu_jit(x_flat_cpu, batch_cpu)
print(time.time() - tic)

tic = time.time()
for i in range(N):
    tmp = fps_cuda_jit(x_flat_gpu, batch_gpu)
print(time.time() - tic)

tic = time.time()
for i in range(N):
    tmp = fps_cpu(x_flat_cpu, batch_cpu)
print(time.time() - tic)

tic = time.time()
for i in range(N):
    tmp = fps_cuda(x_flat_gpu, batch_gpu)
print(time.time() - tic)

tic = time.time()
for i in range(N):
    tmp = fps_native(x_gpu)
print(time.time() - tic)

