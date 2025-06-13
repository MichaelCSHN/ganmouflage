#!/bin/bash

# GANmouflage 项目测试脚本
echo "=== GANmouflage 项目测试开始 ==="
echo "时间: $(date)"
echo ""

# 设置错误处理
set -e

# 激活conda环境
echo "1. 激活conda环境..."
eval "$(conda shell.bash hook)"
conda activate camoGAN_py38pt113cu117
echo "当前环境: $CONDA_DEFAULT_ENV"
echo ""

# 检查Python和PyTorch
echo "2. 检查基础环境..."
echo "Python版本: $(python --version)"
echo "PyTorch版本: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA可用: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q True; then
    echo "CUDA设备: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo ""

# 测试核心依赖
echo "3. 测试核心依赖导入..."
python << 'EOF'
try:
    import torch
    import torchvision
    import numpy as np
    import cv2
    import open3d as o3d
    import trimesh
    import matplotlib.pyplot as plt
    import pandas as pd
    import kornia
    import sklearn
    from skimage import io
    import tensorboardX
    import tqdm
    import lpips
    print("✓ 基础依赖导入成功")
except ImportError as e:
    print(f"✗ 基础依赖导入失败: {e}")
    exit(1)
EOF

# 测试PyTorch3D
echo "4. 测试PyTorch3D..."
python << 'EOF'
try:
    import pytorch3d
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import (
        FoVOrthographicCameras,
        MeshRasterizer,
        RasterizationSettings
    )
    print("✓ PyTorch3D导入成功")
except ImportError as e:
    print(f"✗ PyTorch3D导入失败: {e}")
    exit(1)
EOF

# 测试项目模块
# 修改第5步测试项目模块部分：
echo "5. 测试项目模块..."
python << 'EOF'
try:
    # 测试数据模块
    from data.dataset import CamoDataset_Fake
    from data.scene import Scene
    from data.utils import *
    print("✓ 数据模块导入成功")
    
    # 测试模型模块
    from model_v4.build_models import get_models
    from model_v4.modules.texture_model import TextureNetwork
    from model_v4.modules.sample_normals import sample_normals_v2
    print("✓ 模型模块导入成功")
    
    # 测试训练模块 - 修复导入
    from trainer.basetrainer import BaseTrainer
    from trainer.training import Trainer
    from trainer.random_crop_v3 import RandomResizedCropAroundTarget
    print("✓ 训练模块导入成功")
    
except ImportError as e:
    print(f"✗ 项目模块导入失败: {e}")
    exit(1)
EOF

# 测试配置文件
echo "6. 测试配置文件..."
python << 'EOF'
import yaml
import os

config_dir = "configs"
if os.path.exists(config_dir):
    for config_file in os.listdir(config_dir):
        if config_file.endswith('.yaml'):
            config_path = os.path.join(config_dir, config_file)
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"✓ {config_file} 配置文件有效")
            except Exception as e:
                print(f"✗ {config_file} 配置文件错误: {e}")
else:
    print("⚠ 未找到configs目录")
EOF

# 测试模型构建
# 在第7步测试模型构建部分，修改为：
echo "7. 测试模型构建..."
python << 'EOF'
import yaml
import torch
from model_v4.build_models import get_models

try:
    with open('configs/config_final_4view.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = get_models(config, device)
    
    generator = models['generator']
    discriminator = models['discriminator']
    
    gen_params = sum(p.numel() for p in generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"✓ 模型构建成功")
    print(f"  生成器参数: {gen_params:,}")
    print(f"  判别器参数: {disc_params:,}")
except Exception as e:
    print(f"✗ 模型构建失败: {e}")
EOF

# 测试数据集
echo "8. 测试数据集创建..."
python << 'EOF'
import yaml
import os
from data.dataset import CamoDataset_Fake
from data.scene import Scene

try:
    if os.path.exists('obj4train') and os.listdir('obj4train'):
        # 创建虚拟场景对象
        scene = Scene(
            'scenes/eshmun01/',  # 使用scenes/eshmun01/作为场景目录
            target_size=[256, 384],  # 修改：使用正确的[height, width]格式
            n_views_ref=4,
            n_views_sup=0,
            n_views_val=4,
            distance_to_reference=1.0,
            debug=False,
            cube_scale=1.0,
            cube_save_type=0
        )
        
        # 正确的数据集初始化（移除mode参数）
        dataset = CamoDataset_Fake(
            scenes=[scene],
            rot_limit=[30, 30, 30],
            val=False,
            train_cube_scale_range=[0.8, 1.2]
        )
        print(f"✓ 数据集创建成功，包含 {len(dataset)} 个样本")
    else:
        print("⚠ 跳过数据集测试：未找到训练数据")
except Exception as e:
    print(f"✗ 数据集创建失败: {e}")
    import traceback
    traceback.print_exc()
EOF

# 测试GPU内存
echo "9. 检查GPU状态..."
python << 'EOF'
import torch

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  总内存: {props.total_memory / 1024**3:.1f} GB")
        print(f"  已用内存: {torch.cuda.memory_allocated(i) / 1024**3:.1f} GB")
else:
    print("CUDA不可用，将使用CPU")
EOF

# 快速推理测试
echo "10. 快速推理测试..."

# 显式设置 LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.8/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
echo "LD_LIBRARY_PATH in run_test.sh set to: $LD_LIBRARY_PATH" # 方便调试时查看

python << 'EOF'
import torch
import yaml
from model_v4.build_models import get_models
from pytorch3d.io import load_obj
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

try:
    with open('configs/config_final_4view.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    models = get_models(config, device)
    generator = models['generator']
    generator.eval()
    
    # 正确加载3D模型数据
    verts, faces, _ = load_obj("fake_cube/model_simplified.obj", load_textures=False)
    
    # 修复：正确处理PyTorch3D的Faces对象
    verts_tensor = verts.to(device)
    faces_tensor = faces.verts_idx.to(device)  # 使用verts_idx获取tensor
    
    # 从配置文件获取正确的参数
    batch_size = 1
    n_views = config['data']['scene']['n_views_ref']  # 修复：正确的路径是 data.scene.n_views_ref
    H, W = config['data']['scene']['target_size']  # 修复：正确的路径是 data.scene.target_size
    
    print(f"使用图像尺寸: {H}x{W}")
    print(f"使用视角数: {n_views}")
    
    # 创建相机内参
    cam_K = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(batch_size, n_views, 1, 1).to(device)
    cam_K[:, :, 0, 0] = 500  # fx
    cam_K[:, :, 1, 1] = 500  # fy
    cam_K[:, :, 0, 2] = W/2  # cx
    cam_K[:, :, 1, 2] = H/2  # cy
    
    # 创建相机外参（环绕视角）
    cam_W = torch.zeros(batch_size, n_views, 3, 4).to(device)
    for i in range(n_views):
        angle = i * 2 * 3.14159 / n_views
        cam_W[0, i, 0, 0] = torch.cos(torch.tensor(angle))
        cam_W[0, i, 0, 2] = torch.sin(torch.tensor(angle))
        cam_W[0, i, 1, 1] = 1
        cam_W[0, i, 2, 0] = -torch.sin(torch.tensor(angle))
        cam_W[0, i, 2, 2] = torch.cos(torch.tensor(angle))
        cam_W[0, i, :, 3] = torch.tensor([2, 0, 2])  # 相机位置
    
    # 创建输入数据（使用正确的图像尺寸）
    image_ref = torch.randn(batch_size, n_views, 3, H, W).to(device)
    # 将这一行：
    # background = torch.randn(batch_size, n_views, 3, H, W).to(device)
    # 改为：
    background = torch.randn(batch_size, n_views, H, W, 3).to(device)
    cube_diagonal = torch.tensor([1.0]).to(device)
    depth = torch.ones(batch_size, n_views, H, W).to(device)
    
    # 预热GPU
    print("预热GPU...")
    with torch.no_grad():
        _ = generator(
            depth=depth,
            cam_K=cam_K,
            cam_W=cam_W,
            image_ref=image_ref,
            verts=[verts_tensor] * batch_size,
            faces=[faces_tensor] * batch_size,
            background=background,
            cube_diagonal=cube_diagonal,
            cam_K_ref=cam_K,
            cam_W_ref=cam_W
        )
    
    # 计时推理
    print("开始推理测试...")
    start_time = time.time()
    with torch.no_grad():
        output = generator(
            depth=depth,
            cam_K=cam_K,
            cam_W=cam_W,
            image_ref=image_ref,
            verts=[verts_tensor] * batch_size,
            faces=[faces_tensor] * batch_size,
            background=background,
            cube_diagonal=cube_diagonal,
            cam_K_ref=cam_K,
            cam_W_ref=cam_W
        )
    end_time = time.time()
    
    inference_time = (end_time - start_time) * 1000
    print(f"✓ 推理测试成功")
    print(f"  推理时间: {inference_time:.2f} ms")
    print(f"  输出形状: {output.shape}")
    print(f"  输出类型: {type(output)}")
    
except Exception as e:
    print(f"✗ 推理测试失败: {e}")
    import traceback
    traceback.print_exc()
EOF

echo ""
echo "=== 测试完成 ==="
echo "时间: $(date)"
echo ""
echo "✅ 所有测试已完成！"
echo ""
echo "可用的运行命令:"
echo "  训练模型: bash 02_train_ddp.sh"
echo "  生成2D结果: bash 09a_generate.sh"
echo "  生成3D结果: bash 09b_generate_3D.sh"