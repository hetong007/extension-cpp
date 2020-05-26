from torch.utils.cpp_extension import load
fps_cpu = load(name="fps_cpu", sources=["fps_cpu.cpp"], verbose=True)
help(fps_cpu)
