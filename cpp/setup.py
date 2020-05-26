from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='fps_cpu',
    ext_modules=[
        CppExtension('fps_cpu', ['fps_cpu.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
