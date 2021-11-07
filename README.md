
## Installation
If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`

Since CUDA 11, CUB is shipped with the toolkit. To install the main library, simply run
`pip install .` 
in the root directory.

## Voxel Optimization

See `opt/opt.py`

`sh launch.sh <exp_name> <GPU_id> <data_dir>`

## Evaluation

See `opt/render_imgs.py`

`python render_imgs.py <CHECKPOINT.npz> <data_dir>`

## Automatic hypertuning

See `opt/autotune.py`. Configs in `opt/tasks/*.json`

Automatic eval:
`python autotune.py -g '<space delimited GPU ids>' tasks/eval.json`. Configs in `opt/tasks/*.json`
