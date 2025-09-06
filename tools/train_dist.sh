#!/usr/bin/env sympoint1
WORLD_SIZE=4  
RANK=0  
MASTER_ADDR=localhost  
MASTER_PORT=12355  
export WORLD_SIZE RANK MASTER_ADDR MASTER_PORT

export PYTHONPATH=./
GPUS=1

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
	--dist ./configs/svg/svg_pointT.yaml  \
	--exp_name baseline_nclsw_grelu \
	--sync_bn
