#!/bin/bash

# 显式设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH in train_ddp.sh set to: $LD_LIBRARY_PATH" # 方便调试时查看

export NCCL_DEBUG=INFO

CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node 1 --master_port=2333 \
    train_ddp.py --conf_file=configs/config_final_4view.yaml --scene=mit-16  \
    --log_dir=test_code/mit-16



