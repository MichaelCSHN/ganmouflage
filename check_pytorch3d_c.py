import importlib.util
import os
import pytorch3d # 先导入主模块

print(f"PyTorch3D version: {pytorch3d.__version__}")

# _C.so 文件的确认路径
# (我们已经通过 ls 确认它存在于这个路径)
c_ext_path = "/home/chf/miniconda3/envs/camoGAN_py38pt113cu117/lib/python3.8/site-packages/pytorch3d/_C.cpython-38-x86_64-linux-gnu.so"
print(f"检查 _C.so 文件是否存在于: {c_ext_path}")
print(f"文件实际存在: {os.path.exists(c_ext_path)}")

print("\n尝试使用 importlib 加载 _C.so ...")
try:
    # 尝试将 _C.so 文件作为一个名为 "pytorch3d._C" 的模块加载
    spec = importlib.util.spec_from_file_location("pytorch3d._C", c_ext_path)

    if spec is None:
        print("错误: importlib.util.spec_from_file_location 返回 None，无法加载。")
    else:
        _C_module = importlib.util.module_from_spec(spec)
        if _C_module is None:
            print("错误: importlib.util.module_from_spec 返回 None。")
        else:
            # 这是关键的加载步骤
            spec.loader.exec_module(_C_module) 
            print("成功: pytorch3d._C 已通过 importlib 加载。")
            print(f"_C_module 类型: {type(_C_module)}")

            # 检查函数是否存在
            if hasattr(_C_module, 'point_face_dist_forward'):
                print("函数 point_face_dist_forward 在动态加载的 _C_module 中找到。")
                help(_C_module.point_face_dist_forward)
            else:
                print("错误: 函数 point_face_dist_forward 未在动态加载的 _C_module 中找到。")
                print("可用的属性:", dir(_C_module))

except ImportError as e:
    print(f"来自 importlib 的详细 ImportError: {e}") # 这里会打印更具体的导入错误信息
except AttributeError as e: # 这个 except 块可能不会被触发，因为我们直接用 _C_module
    print(f"CRITICAL ERROR: AttributeError for _C_module: {e}")
except Exception as e:
    print(f"一个预料之外的错误发生: {e}")

print("\n--- 常规导入方式检查 ---")
try:
    # 再次尝试常规方式访问，看重新编译安装后是否有变化
    from pytorch3d import _C as pytorch3d_C_direct
    print(f"通过 'from pytorch3d import _C' 访问 _C 成功: {pytorch3d_C_direct}")
    if hasattr(pytorch3d_C_direct, 'point_face_dist_forward'):
        print("函数 point_face_dist_forward 在常规导入的 _C 中找到。")
        help(pytorch3d_C_direct.point_face_dist_forward)
    else:
        print("错误: 函数 point_face_dist_forward 未在常规导入的 _C 中找到。")
except ImportError as e:
    print(f"通过 'from pytorch3d import _C' 导入失败: {e}")
except AttributeError as e:
    print(f"通过 'pytorch3d._C' 访问失败 (AttributeError): {e}")