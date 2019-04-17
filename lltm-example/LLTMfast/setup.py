from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_fast',
    ext_modules=[
        CUDAExtension('lltm_fast', [
            'lltm_fast.cpp',
            'lltm_fast_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })