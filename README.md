
## Installation
If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`

Since CUDA 11, CUB is shipped with the toolkit. To install the main library, simply run
`pip install .` 
in the root directory.

## Voxel Optimization

See `opt/opt.py`

## Automatic hypertuning

See `opt/autotune.py`. Configs in `opt/tasks/*.json`
