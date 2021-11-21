
## Installation
If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`

Since CUDA 11, CUB is shipped with the toolkit. To install the main library, simply run
`pip install .` 
in the root directory.

## Voxel Optimization

See `opt/opt.py`

`./launch.sh <exp_name> <GPU_id> <data_dir>`

NOTE: can no longer use `sh`

## Evaluation

See `opt/render_imgs.py`

(in opt/)
`python render_imgs.py <CHECKPOINT.npz> <data_dir>`

## Parallel task executor

Including evaluation, ablations, and hypertuning (based on the task_manager one from PlenOctrees)
See `opt/autotune.py`. Configs in `opt/tasks/*.json`

Automatic eval:
`python autotune.py -g '<space delimited GPU ids>' tasks/eval.json`. Configs in `opt/tasks/*.json`

## Using a custom image set

First make sure you have colmap installed. Then

(in opt/)
`bash scripts/proc_colmap.sh <img_dir>`

Where `<img_dir>` should be a directory directly containing png/jpg images from a 
normal perspective camera.
For custom datasets we adopt a data format similar to that in NSVF
<https://github.com/facebookresearch/NSVF>

You should be able to use this dataset directly afterwards. The format will be auto-detected.

To view the data use:
`python scripts/view_data.py <img_dir>`

This should launch a server at localhost:8889

## Random tip: how to make pip install faster

You may notice that this CUDA extension takes forever to install.
A suggestion is using ninja. On Ubuntu,
install it with `sudo apt install ninja-build`.
Then set the environment variable `MAX_JOBS` to the number of CPUS to use in parallel (e.g. 12) in your shell startup script.
This will enable parallel compilation and significantly improve iteration speed.
