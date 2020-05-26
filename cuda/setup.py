from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='fps_cuda',
    ext_modules=[
        CUDAExtension('fps_cuda', [
            'fps_cuda.cpp',
            'fps_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
