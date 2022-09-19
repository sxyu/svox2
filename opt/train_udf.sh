#/bin/bash


# EXP_NAME="udf_dtu_scan63"
# DATA_DIR="../data/dtu/dtu_scan63"
# CONFIG="./configs/udf_dtu.yaml"

EXP_NAME="udf_lego_single_lv"
DATA_DIR="../data/nerf_synthetic/lego"
CONFIG="./configs/udf_syn.yaml"


CKPT_DIR=ckpt/$EXP_NAME


mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

python render_imgs_circle.py $CKPT_DIR $DATA_DIR

