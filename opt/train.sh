#/bin/bash


# EXP_NAME="lego"
# DATA_DIR="../data/nerf_synthetic/lego"
# CONFIG="./configs/syn.json"

# EXP_NAME="horns"
# DATA_DIR="../data/nerf_llff_data/horns"
# CONFIG="./configs/llff.json"

# EXP_NAME="materials"
# DATA_DIR="../data/nerf_synthetic/materials"
# CONFIG="./configs/syn.json"

EXP_NAME="sdf_lego_sphere"
DATA_DIR="../data/nerf_synthetic/lego"
CONFIG="./configs/sdf_syn.json"


CKPT_DIR=ckpt/$EXP_NAME
mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

