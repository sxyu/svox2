from setuptools import setup
import os.path as osp

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('svox2/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []

try:
    ext_modules = [
        CUDAExtension('svox2.csrc', [
            'svox2/csrc/svox2.cpp',
            'svox2/csrc/svox2_kernel.cu',
            'svox2/csrc/render_kernel.cu',
            'svox2/csrc/render_lerp_kernel.cu',
            'svox2/csrc/misc_kernel.cu',
        ], include_dirs=[osp.join(ROOT_DIR, "svox2", "csrc", "include"),],
        optional=True),
    ]
except:
    import warnings
    warnings.warn("Failed to build CUDA extension")
    ext_modules = []

setup(
    name='svox2',
    version=__version__,
    author='Alex Yu',
    author_email='alexyu99126@gmail.com',
    description='PyTorch sparse voxel volume extension, including custom CUDA kernels',
    long_description='PyTorch sparse voxel volume extension, including custom CUDA kernels',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    packages=['svox2', 'svox2.csrc'],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False,
)
