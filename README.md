# $\alpha$Surf



**Note:** This is a preliminary repo for submission of supplementary material only.  




## Setup

First create the virtualenv; we recommend using conda:
```sh
conda env create -f environment.yml
conda activate alphasurf
```

Then install pytorch with CUDA support via:

```sh
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```


Then install the C++/CUDA extension at root by simply running

```
pip install .
```
In the repo root directory.

If your CUDA toolkit is older than 11, then you will need to install CUB as follows:
`conda install -c bottler nvidiacub`.
Since CUDA 11, CUB is shipped with the toolkit.

## Train and eval
An example script for training Plenoxels + our method is avaliable at train_eval.sh



## Random tip (given by authors of Plenoxels): how to make pip install faster for native extensions

You may notice that this CUDA extension takes forever to install.
A suggestion is using ninja. On Ubuntu,
install it with `sudo apt install ninja-build`.
Then set the environment variable `MAX_JOBS` to the number of CPUS to use in parallel (e.g. 12) in your shell startup script.
This will enable parallel compilation and significantly improve iteration speed.
