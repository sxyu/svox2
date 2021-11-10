#!/bin/bash

echo Launching experiment $1
echo GPU $2
echo EXTRA ${@:3}

CKPT_DIR=ckpt/$1
mkdir -p $CKPT_DIR
NOHUP_FILE=$CKPT_DIR/log
echo CKPT $CKPT_DIR
echo LOGFILE $NOHUP_FILE

CUDA_VISIBLE_DEVICES=$2 nohup python -u opt.py -t $CKPT_DIR ${@:3} > $NOHUP_FILE 2>&1 &
echo DETACH
