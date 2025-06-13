#!/bin/bash

# Add CUDA library paths
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# 使用 eshmun01 训练好的模型，在 eshmun02 场景中生成
CUDA_VISIBLE_DEVICES=0 python 09a_generate_result.py --model_path test_code/eshmun02 \
    --out_path test_result/eshmun02 \
    --n 1 \
    --save_background \
    --animals 1 2
