#/bin/bash


# EXP_NAME="udf_dtu_scan63"
# DATA_DIR="../data/dtu/dtu_scan63"
# CONFIG="./configs/udf_dtu.yaml"

EXP_NAME="no_surface_init/2"
DATA_DIR="../data/nerf_synthetic/lego"
CONFIG="./configs/udf_syn.yaml"

# EXP_NAME="norm_loss_exp/_3"
# DATA_DIR="../data/nerf_synthetic/materials"
# CONFIG="./configs/udf_syn.yaml"


CKPT_DIR=ckpt/$EXP_NAME


mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

# python render_imgs_circle.py $CKPT_DIR $DATA_DIR

