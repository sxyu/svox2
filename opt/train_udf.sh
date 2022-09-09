#/bin/bash


EXP_NAME="udf_lego_256_var_reg"
# EXP_NAME="udf_lego_16_mp_ex_lr"
DATA_DIR="../data/nerf_synthetic/lego"
CONFIG="./configs/udf_syn.yaml"
CKPT_DIR=ckpt/$EXP_NAME


mkdir -p $CKPT_DIR

echo CKPT $CKPT_DIR
echo DATA_DIR $DATA_DIR
echo CONFIG $CONFIG

python opt.py -t $CKPT_DIR $DATA_DIR -c $CONFIG

python render_imgs_circle.py $CKPT_DIR $DATA_DIR

