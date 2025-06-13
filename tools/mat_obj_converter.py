import argparse
import scipy.io
import numpy as np
import os
import re

def mat_to_obj(mat_path, obj_path):
    data = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
    # 读取 world_pos 和 faces
    if 'world_pos' not in data or 'faces' not in data:
        raise ValueError(".mat文件中未找到'world_pos'或'faces'变量，请检查文件结构。")
    world_pos = data['world_pos']  # 1xN cell，每个 cell 是3x1
    faces = data['faces']          # 1xM cell，每个 cell 是顶点索引（从1开始）

    # world_pos: cell array to N x 3 numpy array
    vertices = np.array([np.array(pos).flatten() for pos in world_pos]).reshape(-1, 3)
    # faces: cell array to list of lists (索引从1开始，obj同样从1开始)
    faces_list = [np.array(face).flatten() for face in faces]

    with open(obj_path, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces_list:
            # .obj 索引从1开始
            face_str = ' '.join(str(idx) for idx in face)
            f.write(f"f {face_str}\n")

def obj_to_mat(obj_path, mat_path):
    vertices = []
    faces = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.strip().split()[1:])))
            elif line.startswith('f '):
                # 只提取每个面顶点的第一个斜杠前的数字
                tokens = line.strip().split()[1:]
                indices = []
                for t in tokens:
                    # 支持 '1', '1/2', '1//3', '1/2/3' 等格式
                    match = re.match(r'^(\d+)', t)
                    if match:
                        indices.append(int(match.group(1)))
                if indices:
                    faces.append(indices)
    # world_pos: 1xN cell，每个 cell 是3x1
    world_pos = np.empty((1, len(vertices)), dtype=object)
    for i, v in enumerate(vertices):
        world_pos[0, i] = np.array(v).reshape(3, 1)
    # faces: 1xM cell，每个 cell 是 1xK 行向量
    faces_cell = np.empty((1, len(faces)), dtype=object)
    for i, face in enumerate(faces):
        faces_cell[0, i] = np.array(face).reshape(1, -1)
    scipy.io.savemat(mat_path, {'world_pos': world_pos, 'faces': faces_cell})

def main():
    parser = argparse.ArgumentParser(description='Convert between .mat and .obj mesh files.')
    subparsers = parser.add_subparsers(dest='command')

    parser_mat2obj = subparsers.add_parser('mat2obj', help='Convert .mat to .obj')
    parser_mat2obj.add_argument('mat_path', type=str, help='Input .mat file path')
    parser_mat2obj.add_argument('obj_path', type=str, help='Output .obj file path')

    parser_obj2mat = subparsers.add_parser('obj2mat', help='Convert .obj to .mat')
    parser_obj2mat.add_argument('obj_path', type=str, help='Input .obj file path')
    parser_obj2mat.add_argument('mat_path', type=str, help='Output .mat file path')

    args = parser.parse_args()

    if args.command == 'mat2obj':
        mat_to_obj(args.mat_path, args.obj_path)
        print(f"Converted {args.mat_path} to {args.obj_path}")
    elif args.command == 'obj2mat':
        obj_to_mat(args.obj_path, args.mat_path)
        print(f"Converted {args.obj_path} to {args.mat_path}")
    else:
        parser.print_help()

if __name__ == '__main__':
    main() 