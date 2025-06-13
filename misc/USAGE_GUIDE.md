# 集成的2D+3D伪装纹理生成使用指南

## 概述

本指南介绍如何使用集成后的脚本，该脚本结合了原始的2D合成图像生成功能和3D顶点着色功能。脚本会先生成物体在各个相机视图中的合成图像，然后生成带顶点颜色的3D模型。本项目提供了灵活的3D模型顶点颜色生成功能，支持多种模型文件指定方式。

## 功能特性

- **集成工作流**: 先生成2D合成图像，再生成3D顶点着色模型
- **灵活的模型指定**: 支持单个文件、多个文件、目录、混合模式
- **批量处理**: 可以一次处理多个模型文件
- **自动文件发现**: 指定目录时自动查找所有.obj文件
- **智能命名**: 输出文件使用模型文件的原始名称
- **可选功能**: 可以选择跳过2D生成或仅生成2D图像
- **完整输出**: 包括渲染图像、深度掩码、fake views可视化和顶点着色模型

## 快速开始

### 1. 环境激活
```bash
conda activate camoGAN_py38pt113cu117
```

### 2. 使用脚本生成顶点颜色
```bash
bash generate_3D.sh
```

## 配置选项

### 模型文件指定 (OBJ_MODELS)

在 `generate_3D.sh` 中，您可以通过修改 `OBJ_MODELS` 变量来指定要处理的模型文件：

#### 1. 单个模型文件
```bash
OBJ_MODELS="obj4gen/01_level0.obj"
```

#### 2. 多个模型文件（逗号分隔）
```bash
OBJ_MODELS="obj4gen/01_level0.obj,obj4train/00_level0.obj,obj4gen/a.obj"
```

#### 3. 目录中的全部obj文件
```bash
OBJ_MODELS="obj4gen"  # 处理obj4gen目录下的所有.obj文件
# 或者多个目录
OBJ_MODELS="obj4gen,obj4train"  # 处理两个目录下的所有.obj文件
```

#### 4. 模型文件与目录的混合
```bash
OBJ_MODELS="obj4gen,obj4train/00_level0.obj"  # 处理obj4gen目录下所有文件 + 指定的单个文件
```

### 其他配置选项

在脚本中还可以配置以下选项：

- `N_SAMPLES`: 生成样本数量（默认为1）
- `MODEL_PATH`: 训练好的模型权重路径
- `OUTPUT_PATH`: 输出结果保存路径
- `GPU_ID`: 指定使用的GPU编号
- `SAVE_BACKGROUND`: 是否保存背景图像（默认为false）
- `SKIP_2D`: 是否跳过2D图像生成（默认为false）
- `EXPORT_OBJ`: 是否导出3D顶点着色模型（默认为true）
- `EXPORT_TRANSPARENT`: 是否导出透明背景的物体图像（默认为true）

### CUDA内存优化配置

为了解决CUDA内存不足的问题，脚本提供了以下可配置的内存优化参数：

- `QUERY_BATCH_SIZE`: 查询批处理大小（默认为1024）
  - 控制每次查询处理的数据量
  - 内存不足时可以减小此值（如512、256）
  
- `SAFE_BATCH_SIZE`: 安全批处理大小（默认为4）
  - 控制安全处理的批次大小
  - 内存不足时可以减小此值（如2、1）
  
- `CUDA_MAX_SPLIT_SIZE`: CUDA内存分割大小（默认为512MB）
  - 控制CUDA内存分配的最大块大小
  - 内存不足时可以减小此值（如256、128）

#### 内存配置建议

根据您的GPU内存大小，建议使用以下配置：

**8GB GPU内存或更少：**
```bash
QUERY_BATCH_SIZE=256
SAFE_BATCH_SIZE=1
CUDA_MAX_SPLIT_SIZE=128
```

**12GB GPU内存：**
```bash
QUERY_BATCH_SIZE=512
SAFE_BATCH_SIZE=2
CUDA_MAX_SPLIT_SIZE=256
```

**16GB GPU内存或更多：**
```bash
QUERY_BATCH_SIZE=1024
SAFE_BATCH_SIZE=4
CUDA_MAX_SPLIT_SIZE=512
```

#### 配置方法

您可以通过以下两种方式配置这些参数：

1. **直接修改脚本**：编辑 `generate_3D.sh` 文件中的对应变量
2. **设置环境变量**：
   ```bash
   export QUERY_BATCH_SIZE=512
   export SAFE_BATCH_SIZE=2
   export CUDA_MAX_SPLIT_SIZE=256
   bash generate_3D.sh
   ```

## 输出文件

脚本会在指定的输出目录中生成以下文件：

### 2D输出文件
- `sample_0/`: 第一个样本的输出目录
  - `{view_name}_{model_name}.png`: 各个视角的2D合成图像
  - `{view_name}_{model_name}_mask.png`: 对应的深度掩码图像
  - `{view_name}_{model_name}_object_transparent.png`: 透明背景的物体图像（如果启用）
  - `fake_views_{model_name}.png`: fake views可视化（16个视角的网格图）
  - `cube_positions.npy`: 立方体世界坐标位置（如果适用）
- `background/`: 背景图像目录（如果启用保存背景）
  - `background_view{id}.png`: 各个视角的背景图像

### 3D输出文件
- `sample_0/`: 第一个样本的输出目录
  - `{model_name}_vertex_color.obj`: 带顶点颜色的3D模型文件

## 直接使用Python脚本

如果你想直接使用Python脚本而不是shell脚本，可以这样调用：

```bash
# 完整的2D+3D生成
python generate_result_3D.py \
    --model_path "checkpoints/eshmun02" \
    --out_path "result/test_run" \
    --obj_models "obj4gen/01_level0.obj" \
    --export_obj \
    --save_background \
    --n 1

# 仅生成3D顶点着色模型（跳过2D）
python generate_result_3D.py \
    --model_path "checkpoints/eshmun02" \
    --out_path "result/test_run" \
    --obj_models "obj4gen" \
    --skip_2d \
    --export_obj \
    --n 1
```

### 参数说明

- `--model_path`: 模型权重文件夹路径（必需）
- `--out_path`: 输出路径（默认：result/test_run）
- `--obj_models`: 要处理的模型文件或目录
- `--export_obj`: 是否导出带顶点颜色的OBJ模型（默认：True）
- `--save_background`: 是否保存背景图像
- `--skip_2d`: 跳过2D图像生成，仅生成3D顶点着色模型
- `--export_transparent`: 是否导出透明背景的物体图像
- `--n`: 生成样本数量（默认：1）
- `--query_batch_size`: 查询批处理大小（默认：1024）
- `--safe_batch_size`: 安全批处理大小（默认：4）

## 示例

### 示例1：完整的2D+3D生成流程
```bash
# 修改generate_3D.sh中的配置
OBJ_MODELS="obj4gen/01_level0.obj"
SKIP_2D=false
EXPORT_OBJ=true
SAVE_BACKGROUND=true
./generate_3D.sh
```

### 示例2：仅生成3D顶点着色模型
```bash
# 修改generate_3D.sh中的配置
OBJ_MODELS="obj4gen"
SKIP_2D=true
EXPORT_OBJ=true
./generate_3D.sh
```

### 示例3：批量处理多个模型
```bash
# 修改generate_3D.sh中的配置
OBJ_MODELS="obj4gen/01_level0.obj,obj4gen/a.obj,obj4train"
SKIP_2D=false
EXPORT_OBJ=true
./generate_3D.sh
```

### 示例4：仅生成2D图像（不生成3D模型）
```bash
# 修改generate_3D.sh中的配置
OBJ_MODELS="obj4gen"
SKIP_2D=false
EXPORT_OBJ=false
SAVE_BACKGROUND=true
./generate_3D.sh
```

### 示例5：使用CUDA内存优化参数
```bash
# 修改generate_3D.sh中的配置（适用于较小GPU内存）
OBJ_MODELS="obj4gen/01_level0.obj"
QUERY_BATCH_SIZE=256
SAFE_BATCH_SIZE=1
CUDA_MAX_SPLIT_SIZE=128
SKIP_2D=false
EXPORT_OBJ=true
./generate_3D.sh
```

### 示例6：使用环境变量设置内存参数
```bash
# 通过环境变量设置内存优化参数
export QUERY_BATCH_SIZE=512
export SAFE_BATCH_SIZE=2
export CUDA_MAX_SPLIT_SIZE=256
bash generate_3D.sh
```

### 示例7：导出透明背景物体图像
```bash
# 修改generate_3D.sh中的配置
OBJ_MODELS="obj4gen/01_level0.obj"
EXPORT_TRANSPARENT=true
SKIP_2D=false
EXPORT_OBJ=true
./generate_3D.sh
```

### 示例8：使用Python脚本直接导出透明背景图像
```bash
python generate_result_3D.py \
    --model_path "checkpoints/eshmun02" \
    --out_path "result/test_run" \
    --obj_models "obj4gen/01_level0.obj" \
    --export_transparent \
    --n 1
```

## 注意事项

1. 确保指定的文件路径存在且为有效的.obj文件
2. 目录路径应该包含至少一个.obj文件
3. 相对路径将相对于GANmouflage项目根目录解析
4. 生成过程可能需要一些时间，取决于模型复杂度和数量
5. 确保有足够的GPU内存来处理大型模型
6. **CUDA内存优化**：
   - 首次运行时建议使用较小的内存参数，然后根据实际情况调整
   - 不同的GPU型号和驱动版本可能需要不同的内存配置
   - 如果遇到内存不足错误，优先减小 `QUERY_BATCH_SIZE` 和 `SAFE_BATCH_SIZE`
   - 环境变量设置的参数优先级高于脚本中的默认值

## 故障排除

### 常见问题

- 如果遇到"文件不存在"错误，请检查文件路径是否正确
- 确保conda环境已正确激活

### CUDA内存问题

如果遇到CUDA内存不足的错误（如 `CUDA out of memory`），请尝试以下解决方案：

1. **调整内存参数**：根据您的GPU内存大小，修改 `generate_3D.sh` 中的内存优化参数：
   ```bash
   # 对于较小的GPU内存
   QUERY_BATCH_SIZE=256
   SAFE_BATCH_SIZE=1
   CUDA_MAX_SPLIT_SIZE=128
   ```

2. **减少同时处理的模型数量**：如果同时处理多个模型，可以分批处理

3. **清理GPU内存**：
   ```bash
   # 重启Python进程或重新启动终端
   nvidia-smi  # 检查GPU内存使用情况
   ```

4. **检查GPU状态**：
   ```bash
   nvidia-smi  # 查看GPU内存使用情况
   ```

### 性能优化

- 对于大型模型或批量处理，建议使用更大内存的GPU
- 如果处理速度较慢，可以适当增加批处理大小（在内存允许的情况下）
- 定期监控GPU内存使用情况，避免内存泄漏