#/bin/bash


EXP_NAME="plane_lego_t64_h"
DATA_DIR="../data/nerf_synthetic/lego"
CONFIG="./configs/plane_syn.yaml"


CKPT_DIR=ckpt/$EXP_NAME
mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

python render_imgs_circle.py $CKPT_DIR $DATA_DIR

