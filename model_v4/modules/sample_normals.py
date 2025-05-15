import torch
import torch.nn.functional as F_torch # 将原 F 重命名为 F_torch 以免与可能的其他 F 冲突
from pytorch3d import _C             # 直接从 pytorch3d 导入 _C
try:
    import open3d
except ImportError:
    print("Warning! open3d not installed, open3d visualization will fail")

@torch.no_grad()
def sample_normals_v2(verts, faces, p_3d, visualize=False):
    '''
    Args:
        verts: List[Tensor[n_v,3]] # 应该是一个批次中的顶点列表，或者单个顶点张量
        faces: List[Tensor[n_f,3]] # 应该是一个批次中的面列表，或者单个面张量
        p_3d: Tensor[bs, n_render, K, 3] # 需要采样法线的3D点

    Returns:
        surface_normals: Tensor[bs, n_render, K, 3]
    '''
    normals = []
    bs, n_render, K, _ = p_3d.shape
    
    # 确保 verts 和 faces 也是按批次处理的，如果它们是列表的话
    # 假设 len(verts) == bs 和 len(faces) == bs
    for i in range(bs): # 显式按批次循环
        vert_i = verts[i] # 当前批次元素的顶点
        face_i = faces[i] # 当前批次元素的面
        p_i = p_3d[i]     # 当前批次元素的3D点 (n_render, K, 3)

        pt1 = vert_i[face_i[:, 0]]
        pt2 = vert_i[face_i[:, 1]]
        pt3 = vert_i[face_i[:, 2]]  # [n_face_i, 3]

        # 计算每个面的法线
        n_i = torch.cross(pt2 - pt1, pt3 - pt1, dim=1)  # [n_face_i, 3]
        n_i = F_torch.normalize(n_i, p=2, dim=1)      # [n_face_i, 3]

        # p_i 的形状是 (n_render, K, 3)，将其展平为 (n_render*K, 3)
        points_for_dist = p_i.reshape(-1, 3) # 等同于 p_i.flatten(0,1)
        
        # 准备 points_first_idx 和 tris_first_idx (因为我们是单个mesh在C++调用层面)
        points_first_idx = torch.tensor([0], dtype=torch.long, device=p_i.device)
        tris_packed = torch.stack([pt1, pt2, pt3], dim=1) # [n_face_i, 3, 3]
        tris_first_idx = torch.tensor([0], dtype=torch.long, device=p_i.device)
        
        # 第五个参数: max_points，这里是总的点数
        num_query_points = points_for_dist.shape[0] # 即 n_render * K

        # 第六个参数 (float): 通常是最大搜索距离的平方或类似参数。
        # 如果不需要限制，传入一个非常大的值。
        max_sq_dist_for_search = float('inf') 

        # 调用 PyTorch3D 的 C++ 扩展函数
        # 注意：确保这里的 _C 是从 from pytorch3d import _C 正确导入的
        dists_sq, face_idx = _C.point_face_dist_forward(
            points_for_dist,    # arg0: Tensor (P, 3) - 我们查询的点
            points_first_idx,   # arg1: Tensor (1,) - 点的起始索引（对于单个对象）
            tris_packed,        # arg2: Tensor (F, 3, 3) - 三角面片顶点
            tris_first_idx,     # arg3: Tensor (1,) - 面片的起始索引（对于单个对象）
            num_query_points,   # arg4: int - 要处理的点数（与points_for_dist的P对应）
            max_sq_dist_for_search # arg5: float - 新增的参数
        )
        # dists_sq 是平方距离, face_idx 是每个点最近的面索引 (长度为 n_render*K)
        
        # 根据找到的面索引，获取对应的法线
        sampled_n = n_i[face_idx] # [n_render*K, 3]
        normals.append(sampled_n.reshape(n_render, K, 3)) # 恢复形状并添加到列表

    surface_normals = torch.stack(normals, dim=0) # [bs, n_render, K, 3]

    if visualize:
        # 可视化部分的代码，这里保持不变，但请确保open3d已安装且能正常工作
        # 注意：open3d 操作在 CPU 上进行，确保数据已 .cpu()
        for i_vis in range(bs):
            for j_vis in range(n_render):
                pdc = open3d.geometry.PointCloud()
                pdc.points = open3d.utility.Vector3dVector(p_3d[i_vis, j_vis].cpu().numpy())
                pdc.normals = open3d.utility.Vector3dVector(surface_normals[i_vis, j_vis].cpu().numpy())
                
                obj_mesh_vis = open3d.geometry.TriangleMesh(
                    vertices=open3d.utility.Vector3dVector(verts[i_vis].cpu().numpy()),
                    triangles=open3d.utility.Vector3iVector(faces[i_vis].cpu().numpy()))
                mesh_wire_vis = open3d.geometry.LineSet.create_from_triangle_mesh(obj_mesh_vis)
                mesh_frame = open3d.geometry.TriangleMesh.create_coordinate_frame(
                    size=0.2, origin=[0,0,0])
                open3d.visualization.draw_geometries([pdc, mesh_frame, mesh_wire_vis])
                
    return surface_normals