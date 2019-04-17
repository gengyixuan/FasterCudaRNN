from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='lltm_fastseq',
    ext_modules=[
        CUDAExtension('lltm_fastseq', [
            'lltm_fastseq.cpp',
            'lltm_fastseq_kernel.cu',
        ])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })