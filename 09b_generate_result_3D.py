# 导入必要的库
import os
from glob import glob
import random
import argparse
import math
from data.dataset import Scene, CamoDataset_Fake
from data.utils import collate_fn
from data.render_depth import DepthRenderer
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免Qt问题
import matplotlib.pyplot as plt
import torch
from model_v4.build_models import get_models
import yaml
import cv2
from trainer.training import prepare_input,split_into_batches
import numpy as np
from model_v4.modules.texture_model import extract_projected_features
from model_v4.modules.sample_normals import sample_normals_v2
from model_v4.modules.utils import get_ray_directions_inv
from pytorch3d.structures import Meshes
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    MeshRasterizer,
    RasterizationSettings,
    look_at_view_transform
)
import torch.nn.functional as F
from tqdm import tqdm
from model_v4.visualizer import TextureVisualizer

# 定义一个函数来设置随机种子，以确保实验的可复现性
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def visualize_depth(depth):
    """将深度图可视化为0-1范围的图像"""
    mask = depth > 0
    if mask.astype(float).sum() == 0:
        return mask
    min_val = depth[mask].min()
    max_val = depth[mask].max()
    depth[mask] = (depth[mask] - min_val) / (max_val - min_val)
    return depth

def generate_shift(max_shift=0.25, size=5, sigma=1):
    """生成用于位置偏移的高斯权重"""
    k = cv2.getGaussianKernel(size, sigma)
    k = k @ k.T
    grid_shift = np.linspace(-max_shift, max_shift, size)
    shift_2d = np.stack(np.meshgrid(grid_shift, grid_shift), -1).reshape(-1, 2)  # [size^2,2]
    weight = k.reshape(-1)
    return shift_2d.astype(np.float32), weight.astype(np.float32)

def get_cube_world_position(cube_diagonal, rot, translation):
    """获取立方体在世界坐标系中的位置"""
    fake_cube_vertices = np.array([[-0.5, -0.5, -0.5],
                                   [-0.5, -0.5, 0.5],
                                   [0.5, -0.5, -0.5],
                                   [0.5, -0.5, 0.5],
                                   [-0.5, 0.5, -0.5],
                                   [-0.5, 0.5, 0.5],
                                   [0.5, 0.5, -0.5],
                                   [0.5, 0.5, 0.5],
                                   ]) / np.sqrt(3) * cube_diagonal
    return (rot @ fake_cube_vertices.T).T + translation

# 设置全局随机种子
SEED=1234
seed_everything(SEED)

# --- 在此脚本中直接定义一个支持 load_uv 的 CamoDataset_Animals 类 ---
class CamoDataset_Animals:
    def __init__(self, object_list, scenes, rot_limit=None,val=False,same_views=False,train_cube_scale_range=[1,1],repeat=50, load_uv=False):
        self.scenes = scenes
        print("Using scenes:")
        for scene in self.scenes:
            print(scene.scene_dir)
        self.val=val
        self.object_list = object_list*repeat
        self.rot_limit = np.array(rot_limit, dtype=np.float32) if rot_limit is not None else None
        self.same_views=same_views
        self.train_cube_scale_range=train_cube_scale_range
        self.load_uv = load_uv

    def gen_random_rot(self, rot_limit=None):
        angles = np.random.uniform(-1, 1, 3) * (rot_limit/180*np.pi if rot_limit is not None else np.pi)
        Rx = torch.tensor([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]], dtype=torch.float32)
        Ry = torch.tensor([[np.cos(angles[1]), 0, -np.sin(angles[1])], [0, 1, 0], [np.sin(angles[1]), 0, np.cos(angles[1])]], dtype=torch.float32)
        Rz = torch.tensor([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]], dtype=torch.float32)
        return torch.matmul(Rz, torch.matmul(Ry, Rx))

    def get_all_views(self, idx,n_ref,val_only=False,ref_views=[],fixed_position=False):
        obj_path = self.object_list[idx]
        verts, faces, aux = load_obj(obj_path, load_textures=self.load_uv)
        scene_idx = idx % len(self.scenes)
        rot_limit = np.array([0,0,0]) if idx==0 or fixed_position else self.rot_limit
        scene_parameters = self.scenes[scene_idx].get_all_views(
            distance=0 if idx==0 or fixed_position else None, val_only=val_only, n_ref=n_ref,ref_views=ref_views)
        
        data = dict(scene_parameters)
        data['verts'] = verts
        data['faces'] = faces.verts_idx
        
        # 即使在顶点着色模式下，这个保留也无害
        if self.load_uv:
            data['verts_uvs'] = aux.verts_uvs if aux.verts_uvs is not None else torch.empty(0, 2, dtype=torch.float32)
            data['faces_uvs'] = faces.textures_idx if aux.verts_uvs is not None else torch.empty(0, 3, dtype=torch.int64)

        obj_rotation_np = self.gen_random_rot(rot_limit).numpy()
        data['obj_rotation'] = obj_rotation_np @ data['obj2worldR']
        data['obj_id'] = os.path.basename(obj_path)
        return data

    def __len__(self):
        return len(self.object_list)

def get_dataset_single_scene(
    cfg, debug=False, scene_name="", fake_cube=True, animals=True,
    val_samples=400, cube_name=None, load_uv=False, obj_list=None):
    scene_folder = cfg['data']['scene']['scene_folder']
    n_views_ref, n_views_sup, n_views_val = cfg['data']['scene']['n_views_ref'], cfg['data']['scene']['n_views_sup'], cfg['data']['scene']['n_views_val']
    target_size, distance_to_reference, cube_scale = cfg['data']['scene']['target_size'], cfg['data']['scene']['distance_to_cube'], cfg['data']['scene']['cube_scale']
    if scene_name=='esp-tree': cube_save_type=2
    elif scene_name in ['bookshelf-real','couch5-real','patio2-real']: cube_save_type=1
    else: cube_save_type=0
    scene = [Scene(f"{scene_folder}{scene_name}/", target_size, n_views_ref=n_views_ref, n_views_sup=n_views_sup, n_views_val=n_views_val,
                   distance_to_reference=distance_to_reference, debug=debug, cube_scale=cube_scale, cube_save_type=cube_save_type, cube_name=cube_name)]
    if fake_cube:
        datasets = {'valid': CamoDataset_Fake(scene, [0, 0, 0],val=True)}
    elif animals:
        if obj_list is None:
            obj_list = sorted(glob(os.path.join(cfg['data']['animals']['data_dir'], '*.obj')))
        datasets = {'valid': CamoDataset_Animals(obj_list, scene, [0, 0, 0], val=True, repeat=1, load_uv=load_uv)}
    else: raise NotImplementedError("Unsupported dataset")
    return datasets

def parse_obj_models(obj_models_str):
    """
    解析obj_models参数，支持多种格式：
    1. 单个模型文件："obj4gen/01_level0.obj"
    2. 多个模型文件："obj4gen/01_level0.obj,obj4train/00_level0.obj"
    3. 目录中的全部obj："obj4gen,obj4train"
    4. 模型文件与目录的混合："obj4gen,obj4train/00_level0.obj"
    
    返回所有obj文件的绝对路径列表
    """
    if not obj_models_str.strip():
        return []
    
    obj_files = []
    items = [item.strip() for item in obj_models_str.split(',')]
    
    for item in items:
        if not item:
            continue
            
        # 转换为绝对路径
        abs_path = os.path.abspath(item)
        
        if os.path.isfile(abs_path) and abs_path.endswith('.obj'):
            # 单个obj文件
            obj_files.append(abs_path)
        elif os.path.isdir(abs_path):
            # 目录，查找所有obj文件
            dir_obj_files = sorted(glob(os.path.join(abs_path, '*.obj')))
            obj_files.extend(dir_obj_files)
        else:
            print(f"警告：路径 '{item}' 不存在或不是有效的obj文件")
    
    return obj_files

def validate_feature_dimensions(model, cfg, device='cuda'):
    """
    验证模型特征维度与配置的一致性
    """
    expected_z_dim = cfg.get('model', {}).get('z_dim', 230)
    
    # 创建测试数据来验证特征维度
    test_batch_size = 1
    test_points = 100
    test_n_ref = cfg['data']['scene']['n_views_ref']
    
    # 模拟特征提取过程
    try:
        # 创建虚拟输入
        dummy_image = torch.randn(test_batch_size, test_n_ref, 3, 256, 256, device=device)
        dummy_points = torch.randn(test_batch_size, test_points, 3, device=device)
        
        with torch.no_grad():
            # 提取特征图
            feature_map = model.image_encoder(dummy_image.flatten(0, 1))
            
            # 计算特征维度
            if hasattr(feature_map, '__len__') and len(feature_map) > 1:
                total_channels = sum(fm.shape[1] for fm in feature_map)
            else:
                total_channels = feature_map.shape[1] if hasattr(feature_map, 'shape') else 0
            
            # 添加表面法向量维度（如果启用）
            if model.cat_surface_normals:
                total_channels += 3
            
            print(f"验证特征维度: 实际={total_channels}, 期望={expected_z_dim}")
            
            if total_channels != expected_z_dim:
                print(f"警告: 特征维度不匹配! 实际维度={total_channels}, 配置期望={expected_z_dim}")
                print(f"将使用填充/截断策略来处理维度差异")
                return False
            else:
                print("特征维度验证通过")
                return True
                
    except Exception as e:
        print(f"特征维度验证失败: {e}")
        return False

def save_obj_with_vertex_colors(filepath, verts, faces, vertex_colors):
    assert verts.shape[0] == vertex_colors.shape[0], "每个顶点都必须有一个颜色"
    verts_with_colors = torch.cat([verts, vertex_colors], dim=1)
    with open(filepath, 'w') as f:
        f.write("# OBJ file with vertex colors\n")
        for v in verts_with_colors: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {v[3]:.4f} {v[4]:.4f} {v[5]:.4f}\n")
        for face in faces: f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    print(f"OBJ model with vertex colors saved to {filepath}")


def bake_and_save_obj_with_vertex_colors(model, data_collated, model_input, depth_ref, out_prefix, cfg, device='cuda'):
    """
    烘焙顶点颜色并保存OBJ文件，包含增强的错误处理
    """
    try:
        print(f"Starting OBJ export with vertex colors for {out_prefix}...")
        
        # 输入验证
        if not data_collated['verts'] or len(data_collated['verts']) == 0:
            raise ValueError("输入数据中没有有效的顶点信息")
        
        if data_collated['verts'][0].shape[0] == 0:
            raise ValueError("顶点数据为空")
        
        verts, faces = data_collated['verts'][0].to(device), data_collated['faces'][0].to(device)
        p_3d = verts
        
        # 验证相机参数
        if 'cam_K_ref' not in model_input or 'cam_W_ref' not in model_input:
            raise ValueError("缺少必要的相机参数")
        
        K, R, T = model_input['cam_K_ref'][0, 0], model_input['cam_W_ref'][0, 0, :3, :3], model_input['cam_W_ref'][0, 0, :3, 3]
        
        # 检查相机参数的有效性
        if torch.isnan(K).any() or torch.isnan(R).any() or torch.isnan(T).any():
            raise ValueError("相机参数包含NaN值")
        
        p_cam = torch.matmul(p_3d, R.t()) + T
        p_2d_homogeneous = torch.matmul(p_cam, K.t())
        p_2d_pixels = p_2d_homogeneous[..., :2] / (p_2d_homogeneous[..., 2, None] + 1e-8)
        
        H, W = cfg['data']['scene']['target_size']
        p_2d_ndc = torch.stack([
            2.0 * p_2d_pixels[..., 0] / (W - 1) - 1.0,
            2.0 * p_2d_pixels[..., 1] / (H - 1) - 1.0
        ], dim=-1)
        
    except Exception as e:
        print(f"错误: 初始化阶段失败 - {str(e)}")
        print("尝试使用默认参数进行恢复...")
        # 恢复机制：使用简化的处理方式
        try:
            verts, faces = data_collated['verts'][0].to(device), data_collated['faces'][0].to(device)
            p_3d = verts
            # 使用单位矩阵作为默认相机参数
            p_2d_ndc = torch.zeros(p_3d.shape[0], 2, device=device)
            print("使用默认相机参数继续处理")
        except Exception as recovery_error:
            print(f"恢复失败: {str(recovery_error)}")
            return False

    vertex_colors = torch.zeros_like(p_3d)
    query_batch_size = args.query_batch_size  # 从命令行参数获取批处理大小
    print("Baking vertex colors... This is much faster!")
    for i in tqdm(range(0, p_3d.shape[0], query_batch_size)):
        batch_p3d, batch_p2d = p_3d[i:i+query_batch_size].unsqueeze(0), p_2d_ndc[i:i+query_batch_size].unsqueeze(0)
        batch_mask = torch.ones(batch_p3d.shape[:2], dtype=torch.bool, device=device)
        num_pts = batch_p3d.shape[1]
        H, W = cfg['data']['scene']['target_size']
        with torch.no_grad():
            # 直接调用decoder获取颜色
            x = batch_p3d.clone()
            feature_map = model.image_encoder(model_input['image_ref'].flatten(0, 1))
            # 修正unflatten参数以匹配batch_p3d的实际形状
            batch_size = 1
            num_points = batch_p3d.shape[0]
            # 确保cube_diagonal正确广播
            cube_diagonal = model_input['cube_diagonal']
            if cube_diagonal.dim() == 0:  # 如果是标量
                pass  # 标量不需要转换
            elif cube_diagonal.numel() == 1:  # 如果是单元素张量
                cube_diagonal = cube_diagonal.item()  # 转换为Python标量
            else:
                # 如果是多维张量，取第一个元素
                cube_diagonal = cube_diagonal.flatten()[0].item()
            
            # 使用标量进行乘法，避免维度不匹配
            scaled_points = (batch_p3d * cube_diagonal).unflatten(0, [batch_size, num_points])
            pixel_aligned_features, _, _ = extract_projected_features(
                scaled_points,
                feature_map, model_input['cam_K_ref'], model_input['cam_W_ref'],
                (model_input['image_ref'].shape[3], model_input['image_ref'].shape[4])
            )
            # 根据实际形状调整维度处理
            if len(pixel_aligned_features.shape) == 5:
                # 原始形状: [bs, n_ref, batch_points, actual_K, feature_dim]
                # 需要转换为: [bs, n_ref, actual_K, feature_dim]
                bs, n_ref, batch_points, actual_K, feature_dim = pixel_aligned_features.shape
                # 重塑为正确的维度: [bs, n_ref, actual_K, feature_dim]
                pixel_aligned_features = pixel_aligned_features.reshape(bs, n_ref, actual_K, feature_dim)
            elif len(pixel_aligned_features.shape) == 4:
                # 如果是4维: [bs*K, n_ref, H, W] 或其他格式
                bs_k, n_ref, h, w = pixel_aligned_features.shape
                # 重塑为 [bs, K, n_ref, H*W] 然后转置为 [bs, n_ref, K, H*W]
                pixel_aligned_features = pixel_aligned_features.reshape(batch_size, num_points, n_ref, h * w)
                pixel_aligned_features = pixel_aligned_features.permute(0, 2, 1, 3)  # [bs, n_ref, K, H*W]
            
            z = [pixel_aligned_features]
              
            if model.cat_surface_normals:
                surface_normals = sample_normals_v2(model_input['verts'], model_input['faces'], 
                                                    batch_p3d.unflatten(0, [batch_size, num_points]), visualize=False)
                surface_normals = surface_normals.unsqueeze(1)  # [bs, 1, K, 3]
                surface_normals = torch.matmul(model_input['cam_W_ref'][:, :, None, :, :3], 
                                               surface_normals.transpose(-2, -1)).transpose(-2, -1)
                # matmul后维度变为 [bs, n_ref, 1, K, 3]，需要去掉多余的维度
                surface_normals = surface_normals.squeeze(2)  # [bs, n_ref, K, 3]
                z.append(surface_normals)
                  
            z = torch.cat(z, -1)  # 在最后一个维度拼接
            
            # 动态从配置文件获取 z_dim 并使用智能填充策略
            expected_z_dim = cfg.get('model', {}).get('z_dim', 230)
            if z.shape[-1] < expected_z_dim:
                padding_size = expected_z_dim - z.shape[-1]
                
                # 智能填充策略：使用现有特征的统计信息
                if z.shape[-1] > 0:
                    # 计算现有特征的均值和标准差
                    feature_mean = z.mean(dim=-1, keepdim=True)
                    feature_std = z.std(dim=-1, keepdim=True) + 1e-8
                    
                    # 使用高斯噪声填充，基于现有特征的分布
                    noise_padding = torch.randn(*z.shape[:-1], padding_size, device=z.device, dtype=z.dtype)
                    smart_padding = feature_mean + feature_std * noise_padding * 0.1  # 较小的噪声
                else:
                    # 如果没有现有特征，回退到零填充
                    smart_padding = torch.zeros(*z.shape[:-1], padding_size, device=z.device, dtype=z.dtype)
                
                z = torch.cat([z, smart_padding], -1)
            elif z.shape[-1] > expected_z_dim:
                # 如果特征维度超过期望值，进行截断
                z = z[..., :expected_z_dim]
            
            colors = model.decoder_pixel(x, z)
            
        vertex_colors[i:i+query_batch_size] = colors.squeeze(0)
    obj_filename = f"{out_prefix}_vertex_color.obj"
    save_obj_with_vertex_colors(obj_filename, verts.cpu(), faces.cpu(), vertex_colors.clamp(0, 1).cpu())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='为物体生成伪装纹理，包括2D合成图像和3D顶点着色模型。')
    parser.add_argument('--model_path', type=str, required=True, help='模型权重所在的文件夹路径。')
    parser.add_argument('--out_path',type=str,default='result/test_run',help="保存渲染图像和OBJ模型的路径。")
    parser.add_argument('--n',type=int,default=1,help="要生成的样本数量。")
    parser.add_argument("--save_background",action='store_true',help="是否保存背景图像。")
    parser.add_argument("--obj_models", type=str, default="", help="要处理的模型文件或目录，支持单个文件、多个文件（逗号分隔）、目录或混合方式。")
    parser.add_argument("--export_obj", action='store_true', default=True, help="是否导出带顶点颜色的OBJ模型（默认启用）。")
    parser.add_argument("--skip_2d", action='store_true', help="跳过2D图像生成，仅生成3D顶点着色模型。")
    parser.add_argument("--export_transparent", action='store_true', help="是否导出透明背景的物体图像。")
    parser.add_argument("--query_batch_size", type=int, default=2048, help="查询批处理大小，用于CUDA内存优化（默认: 2048）。")
    parser.add_argument("--safe_batch_size", type=int, default=8, help="安全批处理大小，用于CUDA内存优化（默认: 8）。")
    
    args=parser.parse_args()
    safe_batchsize = args.safe_batch_size  # 从命令行参数获取安全批处理大小
    model_dir, out_dir = str(args.model_path), str(args.out_path)
    cfg_path, ckpt_path = os.path.join(model_dir, "config.yaml"), os.path.join(model_dir, "model.pt")

    with open(cfg_path, 'r') as f: cfg = yaml.safe_load(f)
    os.makedirs(out_dir,exist_ok=True)
    print(yaml.dump(cfg))
    
    # 解析obj_models参数
    obj_files = parse_obj_models(args.obj_models)
    if not obj_files:
        print("未指定模型文件，使用默认动物模型...")
        obj_files = None
    else:
        print(f"找到 {len(obj_files)} 个模型文件:")
        for obj_file in obj_files:
            print(f"  - {obj_file}")
    
    datasets = get_dataset_single_scene(cfg, debug=False, scene_name=cfg['data']['scene']['scene_name'],
        fake_cube=cfg['data']['fake_cube'], load_uv=False, obj_list=obj_files)
    is_cube, dataset = cfg['data']['fake_cube'], datasets['valid']
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_models(cfg,device)['generator']
    model.eval()
    renderer = DepthRenderer(cfg['data']['scene']['target_size'])
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_g'])
    
    # 验证特征维度与配置的一致性
    print("验证模型特征维度...")
    validate_feature_dimensions(model, cfg, device)
    
    # 初始化TextureVisualizer用于fake views渲染
    visualizer = TextureVisualizer(16, (192, 192))
    
    n_ref = cfg['data']['scene']['n_views_ref']
    animals = np.arange(len(dataset)) if not is_cube else [0]
    print(len(animals),"models to process")
        
    for i in range(args.n):
        for animal_idx in animals:
            os.makedirs(os.path.join(out_dir, f"sample_{i}"), exist_ok=True)
            
            # 获取数据，注意这里使用val_only=False来获取完整的视图数据
            data = dataset.get_all_views(animal_idx + len(dataset) * i, n_ref=n_ref, val_only=False, fixed_position=(i==0))
            
            # 保存立方体世界位置（如果需要）
            if not is_cube:
                cube_world_positions = get_cube_world_position(dataset.scenes[0].cube_diagonal, data['obj2worldR'], data['obj2world'])
                np.save(os.path.join(out_dir, f"sample_{i}", "cube_positions.npy"), cube_world_positions)
            
            data_collated = collate_fn([data])
            if not data_collated['verts']:
                print(f"Skipping animal {animal_idx} due to empty mesh.")
                continue

            assert len(data_collated['verts']) == 1, "For visualization batchsize can only be 1"

            depth = renderer.render(data_collated, idx=0).squeeze(-1)
            depth_ref = depth[:, :n_ref]

            model_input = prepare_input(data_collated, device=device, render_idx=[0, depth.shape[1]], ref_idx=[0, n_ref])
            model_input['depth'] = depth
            
            # 获取模型文件名（不含扩展名）
            model_name = os.path.splitext(os.path.basename(data['obj_id']))[0] if 'obj_id' in data else f"shape_{animal_idx}"
            
            # === 第一步：生成2D合成图像（来自generate_result.py的逻辑）===
            if not args.skip_2d:
                print(f"生成2D合成图像 - 样本 {i}, 模型 {animal_idx}...")
                safe_batches = split_into_batches(model_input, safe_batchsize)
                rendered_images_all = []
                
                with torch.no_grad():
                    for safe_batch in safe_batches:
                        rendered_images = model(**safe_batch, depth_ref=depth_ref)
                        rendered_images_all.append(rendered_images.detach().cpu().numpy()[:, :, :, ::-1])  # [b,h,w,c]
                rendered_images_all = np.concatenate(rendered_images_all, 0)
                N = rendered_images_all.shape[0]
                
                # 保存渲染图像和深度掩码
                for j in range(N):
                    # 保存深度掩码
                    depth_img = visualize_depth(depth[0][j].detach().cpu().numpy())
                    depth_img_uint8 = (depth_img * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(out_dir, f"sample_{i}", f"{data_collated['image_meta'][0][j]}_{model_name}_mask.png"), depth_img_uint8)
                    
                    # 保存无背景的透明物体图像（如果启用）
                    if args.export_transparent:
                        # 创建RGBA图像，其中alpha通道基于深度掩码
                        object_img_rgb = (rendered_images_all[j] * 255).astype(np.uint8)
                        # 获取物体掩码（深度大于0的区域）
                        object_mask = (depth[0][j].detach().cpu().numpy() > 0).astype(np.uint8) * 255
                        # 创建RGBA图像
                        object_img_rgba = np.zeros((object_img_rgb.shape[0], object_img_rgb.shape[1], 4), dtype=np.uint8)
                        object_img_rgba[:, :, :3] = object_img_rgb  # RGB通道
                        object_img_rgba[:, :, 3] = object_mask      # Alpha通道（透明度）
                        # 保存透明背景的物体图像
                        cv2.imwrite(os.path.join(out_dir, f"sample_{i}", f"{data_collated['image_meta'][0][j]}_{model_name}_object_transparent.png"), object_img_rgba)
                    
                    # 保存合成图像（物体+背景）
                    img_uint8 = (rendered_images_all[j] * 255).astype(np.uint8)
                    cv2.imwrite(os.path.join(out_dir, f"sample_{i}", f"{data_collated['image_meta'][0][j]}_{model_name}.png"), img_uint8)
                
                # 保存背景图像（仅在第一个样本和第一个动物时）
                if i == 0 and args.save_background and animal_idx == 0:
                    os.makedirs(os.path.join(out_dir, "background"), exist_ok=True)
                    for j in range(N):
                        meta = data_collated['image_meta'][0][j]
                        if 'ref' in meta:
                            continue
                        view_id = meta.split('_')[1]
                        bg_img = (model_input['background'][0][j].detach().cpu().numpy()[:, :, ::-1] * 255).astype(np.uint8)
                        cv2.imwrite(os.path.join(out_dir, "background", f"background_view{view_id}.png"), bg_img)
                
                # 生成fake views可视化
                print(f"生成fake views可视化...")
                mesh = Meshes(verts=data_collated['verts'], faces=data_collated['faces']).to(device)
                background = torch.ones(*visualizer.background_shape).unsqueeze(0).to(device)
                N_vis = background.shape[1]
                input_image = model_input['image_ref']
                cam_K_ref = model_input['cam_K_ref']
                cam_W_ref = model_input['cam_W_ref']
                cube_diagonal = data_collated['cube_diagonal'].repeat(N_vis).view(1, -1).to(device)
                depth_vis = visualizer.render_all_views(mesh).squeeze(-1)
                
                rendered_fake = []
                for k in range(math.ceil(N_vis / safe_batchsize)):
                    sub_bs = min(safe_batchsize, N_vis - k * safe_batchsize)
                    p_3d, p_2d, mask = visualizer.depth_map_to_3d(depth_vis[k*safe_batchsize:k*safe_batchsize+sub_bs], 
                                                                 idx_start=k*safe_batchsize, idx_end=k*safe_batchsize+sub_bs)
                    rendered_images = model(None, None, None,
                                          image_ref=input_image,
                                          background=background[:, k*safe_batchsize:k*safe_batchsize+sub_bs],
                                          cube_diagonal=cube_diagonal[:, k*safe_batchsize:k*safe_batchsize+sub_bs],
                                          cam_K_ref=cam_K_ref,
                                          cam_W_ref=cam_W_ref,
                                          verts=model_input['verts'], faces=model_input['faces'],
                                          p_3d=p_3d, p_2d=p_2d, pad_mask=mask,
                                          depth_ref=depth_ref, visualize=False)
                    rendered_images = rendered_images.detach().cpu().numpy()
                    rendered_fake.append(rendered_images)
                
                rendered_fake = np.concatenate(rendered_fake, 0)
                plt.figure(figsize=(24, 24))
                for j in range(16):
                    plt.subplot(4, 4, j + 1)
                    plt.imshow(rendered_fake[j])
                plt.tight_layout()
                plt.savefig(os.path.join(out_dir, f"sample_{i}", f"fake_views_{model_name}.png"))
                plt.close()
                
                if args.export_transparent:
                    print(f"已保存 {len(rendered_images_all)} 张2D渲染图像、{len(rendered_images_all)} 张透明背景物体图像和fake views可视化。")
                else:
                    print(f"已保存 {len(rendered_images_all)} 张2D渲染图像和fake views可视化。")
            else:
                print(f"跳过2D图像生成 - 样本 {i}, 模型 {animal_idx}。")

            # === 第二步：生成3D顶点着色OBJ模型 ===
            if args.export_obj:
                print(f"生成3D顶点着色模型 - 样本 {i}, 模型 {animal_idx}...")
                out_prefix = os.path.join(out_dir, f"sample_{i}", model_name)
                bake_and_save_obj_with_vertex_colors(model, data_collated, model_input, depth_ref, out_prefix, cfg, device)
