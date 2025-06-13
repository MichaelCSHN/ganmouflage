import open3d as o3d
import numpy as np
import os
import glob

def generate_point_cloud(obj_path, npy_path):
    # 读取 OBJ 文件
    mesh = o3d.io.read_triangle_mesh(obj_path)
    
    # 生成点云
    pcd = mesh.sample_points_uniformly(number_of_points=10000)
    
    # 计算法线
    pcd.estimate_normals()
    
    # 保存为 NPY 文件
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    point_cloud = np.concatenate([points, normals], axis=1)
    np.save(npy_path, point_cloud)

if __name__ == "__main__":
    # 处理 obj4train 目录下的所有 OBJ 文件
    obj_files = glob.glob("obj4train/*.obj")
    for obj_file in obj_files:
        npy_file = obj_file.replace(".obj", ".npy")
        print(f"Processing {obj_file} -> {npy_file}")
        generate_point_cloud(obj_file, npy_file) 