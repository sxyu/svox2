#!/bin/bash

# USAGE: bash proc_colmap.sh <dir of images>

python run_colmap.py $1 ${@:2}
python colmap2nsvf.py $1/sparse/0
python create_split.py -y $1
