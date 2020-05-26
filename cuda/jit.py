from torch.utils.cpp_extension import load
fps_cuda = load(
    'fps_cuda', ['fps_cuda.cpp', 'fps_cuda_kernel.cu'], verbose=True)
help(fps_cuda)
