from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cache_lstm',
    ext_modules=[
        CUDAExtension('cache_lstm', [
            'cacheLSTM.cpp',
            'cacheLSTM_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })