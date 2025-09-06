#!/usr/bin/env sympoint
WORLD_SIZE=4
RANK=0
MASTER_ADDR=localhost
MASTER_PORT=12355
export WORLD_SIZE RANK MASTER_ADDR MASTER_PORT

export PYTHONPATH=./
GPUS=1

OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/test.py \
	 ./configs/svg/svg_pointT.yaml  ./best.pth --dist
