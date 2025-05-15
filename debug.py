# import pytorch3d
# import inspect

# # 1. 查看 PyTorch3D 模块的安装路径
# print(f"PyTorch3D module location: {pytorch3d.__file__}")

# # 2. 查看 _C 扩展模块的路径
# print(f"PyTorch3D _C extension location: {pytorch3d._C.__file__}")

# # 3. 查看 point_face_dist_forward 函数的帮助文档/签名
# # 这通常会显示 C++ 绑定的函数签名
# help(pytorch3d._C.point_face_dist_forward)

# # 4. (如果 help 输出不够清晰) 尝试使用 inspect 模块 (可能对C扩展效果有限)
# # try:
# #     print(inspect.signature(pytorch3d._C.point_face_dist_forward))
# # except ValueError:
# #     print("inspect.signature could not determine the signature for the C extension.")


import pytorch3d
print(f"PyTorch3D version: {pytorch3d.__version__}") # 再次确认版本号能被访问

# 尝试直接访问 _C 模块
try:
    print(f"Attempting to access pytorch3d._C: {pytorch3d._C}")
    print(f"pytorch3d._C type: {type(pytorch3d._C)}")
    if hasattr(pytorch3d._C, '__file__'):
        print(f"pytorch3d._C location: {pytorch3d._C.__file__}")
    else:
        print("pytorch3d._C does not have a __file__ attribute (might be a built-in module style).")
    
    # 检查是否存在 point_face_dist_forward 函数
    if hasattr(pytorch3d._C, 'point_face_dist_forward'):
        print("pytorch3d._C.point_face_dist_forward exists.")
        help(pytorch3d._C.point_face_dist_forward) # 查看其帮助信息和签名
    else:
        print("pytorch3d._C.point_face_dist_forward NOT found.")
        print("Available attributes in pytorch3d._C:", dir(pytorch3d._C))

except AttributeError as e:
    print(f"AttributeError when accessing pytorch3d._C: {e}")
except ImportError as e:
    print(f"ImportError related to pytorch3d._C: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")