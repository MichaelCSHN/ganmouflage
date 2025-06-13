#!/bin/bash

# 集成的2D+3D伪装纹理生成脚本
# 先生成2D合成图像，然后生成3D顶点着色模型
# 支持指定单个文件、多个文件、目录或混合方式

# ==============================================================================
# 1. 设置环境 (Setup Environment)
# ==============================================================================
# 确保PyTorch可以找到CUDA库。这对于在Conda或WSL环境中运行尤其重要。
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"

# 设置CUDA内存管理，避免内存碎片化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:$CUDA_MAX_SPLIT_SIZE
echo "PYTORCH_CUDA_ALLOC_CONF set to: $PYTORCH_CUDA_ALLOC_CONF"

# ==============================================================================
# 2. 配置参数 (Configuration Parameters)
# ==============================================================================
MODEL_PATH="test_code/eshmun02"     # 模型权重文件夹路径
OUT_PATH="test_result/eshmun02"     # 输出路径
N_SAMPLES=1                         # 生成样本数量

# CUDA内存优化参数 (CUDA Memory Optimization Parameters)
# 用户可以根据GPU内存大小调整这些参数
QUERY_BATCH_SIZE=2048               # 查询批处理大小 (默认: 2048, 原始: 8192)
SAFE_BATCH_SIZE=8                   # 安全批处理大小 (默认: 8, 原始: 32)
CUDA_MAX_SPLIT_SIZE=512             # CUDA最大分割大小MB (默认: 512)

# 要处理的OBJ模型文件或目录
# 支持以下格式：
# 1. 单个文件: "path/to/model.obj"
# 2. 多个文件: "path/to/model1.obj,path/to/model2.obj"
# 3. 目录: "path/to/directory" (处理目录下所有.obj文件)
# 4. 混合: "path/to/directory,path/to/specific.obj"
OBJ_MODELS="obj4gen/simple_geoZY.obj" 
# 其他选项
SAVE_BACKGROUND=false               # 是否保存背景图像
SKIP_2D=false                       # 是否跳过2D图像生成
EXPORT_OBJ=true                     # 是否导出3D顶点着色模型
EXPORT_TRANSPARENT=true             # 是否导出透明背景的物体图像

# ==============================================================================
# 执行脚本
# ==============================================================================
echo "启动集成的2D+3D伪装纹理生成..."
echo "模型路径: $MODEL_PATH"
echo "输出路径: $OUT_PATH"
echo "OBJ模型: $OBJ_MODELS"
echo "样本数量: $N_SAMPLES"
echo "跳过2D生成: $SKIP_2D"
echo "导出3D模型: $EXPORT_OBJ"
echo "导出透明背景物体图像: $EXPORT_TRANSPARENT"
echo "CUDA内存优化参数:"
echo "  查询批处理大小: $QUERY_BATCH_SIZE"
echo "  安全批处理大小: $SAFE_BATCH_SIZE"
echo "  CUDA最大分割大小: ${CUDA_MAX_SPLIT_SIZE}MB"

# 构建命令参数
CMD_ARGS="--model_path \"$MODEL_PATH\" --out_path \"$OUT_PATH\" --obj_models \"$OBJ_MODELS\" --n $N_SAMPLES"
CMD_ARGS="$CMD_ARGS --query_batch_size $QUERY_BATCH_SIZE --safe_batch_size $SAFE_BATCH_SIZE"

if [ "$SAVE_BACKGROUND" = true ]; then
    CMD_ARGS="$CMD_ARGS --save_background"
fi

if [ "$SKIP_2D" = true ]; then
    CMD_ARGS="$CMD_ARGS --skip_2d"
fi

if [ "$EXPORT_OBJ" = true ]; then
    CMD_ARGS="$CMD_ARGS --export_obj"
fi

if [ "$EXPORT_TRANSPARENT" = true ]; then
    CMD_ARGS="$CMD_ARGS --export_transparent"
fi

# 运行Python脚本
echo "执行命令: python 09b_generate_result_3D.py $CMD_ARGS"
eval "python 09b_generate_result_3D.py $CMD_ARGS"

echo "集成的2D+3D伪装纹理生成完成！"
