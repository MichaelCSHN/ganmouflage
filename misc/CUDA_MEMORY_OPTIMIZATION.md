# CUDA内存优化参数配置指南

## 概述

为了解决CUDA内存不足的问题，我们在 `generate_3D.sh` 脚本中添加了可配置的CUDA内存优化参数。用户可以根据自己的GPU内存大小调整这些参数。

## 可配置参数

### 1. QUERY_BATCH_SIZE（查询批处理大小）
- **默认值**: 2048
- **原始值**: 8192
- **说明**: 控制3D顶点颜色烘焙时的批处理大小
- **建议**: 
  - GPU内存 >= 8GB: 可使用 4096 或更高
  - GPU内存 4-8GB: 使用 2048（默认）
  - GPU内存 < 4GB: 使用 1024 或更低

### 2. SAFE_BATCH_SIZE（安全批处理大小）
- **默认值**: 8
- **原始值**: 32
- **说明**: 控制2D图像渲染时的批处理大小
- **建议**:
  - GPU内存 >= 8GB: 可使用 16 或更高
  - GPU内存 4-8GB: 使用 8（默认）
  - GPU内存 < 4GB: 使用 4 或更低

### 3. CUDA_MAX_SPLIT_SIZE（CUDA最大分割大小）
- **默认值**: 512 MB
- **说明**: 控制CUDA内存分配器的最大分割大小，避免内存碎片化
- **建议**:
  - GPU内存 >= 8GB: 可使用 1024
  - GPU内存 4-8GB: 使用 512（默认）
  - GPU内存 < 4GB: 使用 256

## 使用方法

### 方法1: 直接修改脚本

编辑 `generate_3D.sh` 文件，修改以下参数：

```bash
# CUDA内存优化参数 (CUDA Memory Optimization Parameters)
# 用户可以根据GPU内存大小调整这些参数
QUERY_BATCH_SIZE=2048               # 查询批处理大小
SAFE_BATCH_SIZE=8                   # 安全批处理大小
CUDA_MAX_SPLIT_SIZE=512             # CUDA最大分割大小MB
```

### 方法2: 临时设置环境变量

在运行脚本前设置环境变量：

```bash
# 设置较小的批处理大小（适用于低内存GPU）
export QUERY_BATCH_SIZE=1024
export SAFE_BATCH_SIZE=4
export CUDA_MAX_SPLIT_SIZE=256

# 然后运行脚本
bash generate_3D.sh
```

## 内存使用建议

### 低内存GPU（< 4GB）
```bash
QUERY_BATCH_SIZE=1024
SAFE_BATCH_SIZE=4
CUDA_MAX_SPLIT_SIZE=256
```

### 中等内存GPU（4-8GB）
```bash
QUERY_BATCH_SIZE=2048    # 默认值
SAFE_BATCH_SIZE=8        # 默认值
CUDA_MAX_SPLIT_SIZE=512  # 默认值
```

### 高内存GPU（>= 8GB）
```bash
QUERY_BATCH_SIZE=4096
SAFE_BATCH_SIZE=16
CUDA_MAX_SPLIT_SIZE=1024
```

## 故障排除

### 如果仍然遇到CUDA内存不足错误：

1. **进一步减小批处理大小**：
   ```bash
   QUERY_BATCH_SIZE=512
   SAFE_BATCH_SIZE=2
   ```

2. **检查GPU内存使用情况**：
   ```bash
   nvidia-smi
   ```

3. **关闭其他占用GPU内存的程序**

4. **考虑使用CPU模式**（如果支持）

### 如果处理速度太慢：

1. **适当增加批处理大小**（在内存允许的情况下）
2. **检查是否有足够的GPU内存空间**

## 参数验证

使用测试脚本验证参数传递：

```bash
bash test_cuda_params.sh
```

这将显示参数是否正确传递给Python脚本，而不实际运行生成过程。

## 注意事项

- 较小的批处理大小会降低处理速度，但减少内存使用
- 较大的批处理大小会提高处理速度，但需要更多内存
- 建议从较小的值开始，逐步增加直到找到最适合您GPU的配置
- 这些参数的最佳值取决于您的具体GPU型号和可用内存