import os
import numpy as np
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance
import glob

def verify_npy_file(npy_path):
    """验证.npy文件的格式是否正确"""
    try:
        data = np.load(npy_path)
        if data.shape != (2048, 6):
            print(f"错误: {npy_path} 的形状不正确，期望 (2048, 6)，实际 {data.shape}")
            return False
        if data.dtype != np.float64:
            print(f"错误: {npy_path} 的数据类型不正确，期望 float64，实际 {data.dtype}")
            return False
        if np.isnan(data).any():
            print(f"错误: {npy_path} 包含 NaN 值")
            return False
        if np.isinf(data).any():
            print(f"错误: {npy_path} 包含 Inf 值")
            return False
        return True
    except Exception as e:
        print(f"验证 {npy_path} 时出错: {str(e)}")
        return False

def process_obj_to_npy(obj_path, npy_path):
    """处理单个obj文件，生成对应的npy文件"""
    try:
        # 读取obj文件
        with open(obj_path, 'r') as f:
            lines = f.readlines()
        
        # 解析顶点和法线
        vertices = []
        normals = []
        current_normal = None
        
        # 检查并修复顶点格式
        fixed_lines = []
        for line in lines:
            if line.startswith('vn'):
                # 保持法线不变
                fixed_lines.append(line)
                parts = line.strip().split()[1:]
                current_normal = [float(x) for x in parts]
            elif line.startswith('v'):
                # 检查顶点行是否包含颜色信息
                parts = line.strip().split()
                if len(parts) == 4:  # 只有坐标
                    # 添加颜色信息
                    fixed_line = line.strip() + " 1.000000 1.000000 1.000000\n"
                    fixed_lines.append(fixed_line)
                    vertices.append([float(x) for x in parts[1:4]])
                else:  # 已有颜色信息
                    fixed_lines.append(line)
                    vertices.append([float(x) for x in parts[1:4]])
                if current_normal is not None:
                    normals.append(current_normal)
            else:
                fixed_lines.append(line)
        
        # 保存修复后的obj文件
        with open(obj_path, 'w') as f:
            f.writelines(fixed_lines)
        
        # 转换为numpy数组
        vertices = np.array(vertices, dtype=np.float64)
        normals = np.array(normals, dtype=np.float64)
        
        # 确保顶点和法线数量匹配
        assert len(vertices) == len(normals), f"顶点和法线数量不匹配: {len(vertices)} vs {len(normals)}"
        
        # 随机采样2048个点
        if len(vertices) > 2048:
            indices = np.random.choice(len(vertices), 2048, replace=False)
            vertices = vertices[indices]
            normals = normals[indices]
        else:
            # 如果点数不足，重复采样
            indices = np.random.choice(len(vertices), 2048, replace=True)
            vertices = vertices[indices]
            normals = normals[indices]
        
        # 合并点和法线
        combined = np.concatenate([vertices, normals], axis=1)
        
        # 保存为npy文件
        np.save(npy_path, combined)
        print(f"成功处理 {obj_path} -> {npy_path}")
        print(f"数据形状: {combined.shape}, 数据类型: {combined.dtype}")
        return True
        
    except Exception as e:
        print(f"处理 {obj_path} 时出错: {str(e)}")
        return False

def verify_npy_files(npy_dir):
    """验证所有npy文件"""
    npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
    valid_files = []
    invalid_files = []
    
    for npy_file in npy_files:
        try:
            data = np.load(npy_file)
            if data.shape == (2048, 6) and data.dtype == np.float64:
                valid_files.append(npy_file)
            else:
                print(f"文件 {npy_file} 格式不正确:")
                print(f"形状: {data.shape}, 期望: (2048, 6)")
                print(f"类型: {data.dtype}, 期望: float64")
                invalid_files.append(npy_file)
        except Exception as e:
            print(f"验证 {npy_file} 时出错: {str(e)}")
            invalid_files.append(npy_file)
    
    return valid_files, invalid_files

def main(dir):
    # 处理obj4train目录下的所有obj文件
    obj_dir = dir
    npy_dir = dir
    
    # 确保输出目录存在
    os.makedirs(npy_dir, exist_ok=True)
    
    # 获取所有obj文件
    obj_files = glob.glob(os.path.join(obj_dir, "*.obj"))
    
    # 处理每个obj文件
    success_count = 0
    fail_count = 0
    
    for obj_file in obj_files:
        npy_file = os.path.join(npy_dir, os.path.splitext(os.path.basename(obj_file))[0] + ".npy")
        if process_obj_to_npy(obj_file, npy_file):
            success_count += 1
        else:
            fail_count += 1
    
    print("\n处理完成:")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    
    # 验证所有npy文件
    print("\n验证所有npy文件:")
    valid_files, invalid_files = verify_npy_files(npy_dir)
    print("验证结果:")
    print(f"有效: {len(valid_files)} 个文件")
    print(f"无效: {len(invalid_files)} 个文件")

if __name__ == "__main__":
    main("obj4train")