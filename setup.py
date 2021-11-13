from setuptools import setup
import os
import os.path as osp
import warnings

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT_DIR = osp.dirname(osp.abspath(__file__))

__version__ = None
exec(open('svox2/version.py', 'r').read())

CUDA_FLAGS = []
INSTALL_REQUIREMENTS = []
include_dirs = [osp.join(ROOT_DIR, "svox2", "csrc", "include")]

# From PyTorch3D
cub_home = os.environ.get("CUB_HOME", None)
if cub_home is None:
	prefix = os.environ.get("CONDA_PREFIX", None)
	if prefix is not None and os.path.isdir(prefix + "/include/cub"):
		cub_home = prefix + "/include"

if cub_home is None:
	warnings.warn(
		"The environment variable `CUB_HOME` was not found."
		"Installation will fail if your system CUDA toolkit version is less than 11."
		"NVIDIA CUB can be downloaded "
		"from `https://github.com/NVIDIA/cub/releases`. You can unpack "
		"it to a location of your choice and set the environment variable "
		"`CUB_HOME` to the folder containing the `CMakeListst.txt` file."
	)
else:
	include_dirs.append(os.path.realpath(cub_home).replace("\\ ", " "))

try:
    ext_modules = [
        CUDAExtension('svox2.csrc', [
            'svox2/csrc/svox2.cpp',
            'svox2/csrc/svox2_kernel.cu',
            'svox2/csrc/render_lerp_kernel_cuvol.cu',
            'svox2/csrc/render_lerp_kernel_nvol.cu',
            'svox2/csrc/render_svox1_kernel.cu',
            'svox2/csrc/misc_kernel.cu',
            'svox2/csrc/loss_kernel.cu',
            'svox2/csrc/optim_kernel.cu',
        ], include_dirs=include_dirs,
        optional=False),
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
