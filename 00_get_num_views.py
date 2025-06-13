import os
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
import subprocess
import shutil # Added for file copying

# files=glob.glob("scenes/*/good_cams.txt") # Original line
scene_base_dirs = glob.glob("scenes/*/") # Get all potential scene directories
files = [] # This will store paths to good_cams.txt to be processed

for scene_dir_path in scene_base_dirs:
    if not os.path.isdir(scene_dir_path): # Ensure it's a directory
        continue

    scene_name = os.path.basename(os.path.normpath(scene_dir_path)) # Get scene name like Gearth01
    good_cams_file = os.path.join(scene_dir_path, "good_cams.txt")
    list_txt_file = os.path.join(scene_dir_path, "list.txt")
    bundle_out_list_txt_file = os.path.join(scene_dir_path, "bundle.out.list.txt")

    source_file_to_copy = None
    current_good_cams_path_for_scene = None # Path to the good_cams.txt for *this* scene, if found/created

    if os.path.isfile(good_cams_file):
        print(f"Found existing good_cams.txt for scene {scene_name} at {good_cams_file}")
        files.append(good_cams_file)
        current_good_cams_path_for_scene = good_cams_file
    else:
        if os.path.isfile(list_txt_file):
            print(f"Found list.txt for scene {scene_name} at {list_txt_file}. Will copy to {good_cams_file}")
            source_file_to_copy = list_txt_file
        elif os.path.isfile(bundle_out_list_txt_file):
            print(f"Found bundle.out.list.txt for scene {scene_name} at {bundle_out_list_txt_file}. Will copy to {good_cams_file}")
            source_file_to_copy = bundle_out_list_txt_file
        
        if source_file_to_copy:
            try:
                shutil.copyfile(source_file_to_copy, good_cams_file)
                print(f"Successfully copied {source_file_to_copy} to {good_cams_file} for scene {scene_name}")
                files.append(good_cams_file)
                current_good_cams_path_for_scene = good_cams_file
            except Exception as e:
                print(f"Error copying {source_file_to_copy} to {good_cams_file} for scene {scene_name}: {e}")
        else:
            print(f"No good_cams.txt, list.txt, or bundle.out.list.txt found for scene {scene_name} in {scene_dir_path}")

    # Reformat the good_cams.txt for the current scene if it was found or created
    if current_good_cams_path_for_scene:
        try:
            with open(current_good_cams_path_for_scene, 'r') as f_read:
                lines = f_read.readlines()
            
            transformed_lines = []
            made_change = False
            for line in lines:
                original_line_content = line.strip()
                # Handle case where line contains '\\n' (escaped) or '\n' (as text)
                split_candidates = []
                if '\\n' in original_line_content:
                    split_candidates = original_line_content.split('\\n')
                elif '\n' in original_line_content:
                    split_candidates = original_line_content.split('\n')
                else:
                    split_candidates = [original_line_content]
                for candidate in split_candidates:
                    candidate = candidate.strip()
                    if not candidate:
                        continue
                    new_line_content = candidate.replace("view", "").replace(".jpg", "").replace(".png", "")
                    try:
                        number = int(new_line_content)
                        new_line_content = str(number)
                        if new_line_content != candidate:
                            made_change = True
                        transformed_lines.append(new_line_content)
                    except ValueError:
                        print(f"Warning: Skipping invalid line in {current_good_cams_path_for_scene}: {candidate}")
                        continue
            
            if made_change:
                with open(current_good_cams_path_for_scene, 'w', newline='\n') as f_write:
                    for line in transformed_lines:
                        f_write.write(line + '\n')
                print(f"Reformatted content of {current_good_cams_path_for_scene}")
            # else: # Optional: print if no changes were made
            #    print(f"Content of {current_good_cams_path_for_scene} is already in target format or no relevant patterns found.")

        except Exception as e:
            print(f"Error reformatting content of {current_good_cams_path_for_scene}: {e}")

result=[]
for file in files:
    with open(file,'r') as f:
        good_views=f.readlines()
    #if len(good_views)<9:
    #    print(file,", num good views:",len(good_views))
    seed= sum([ord(x) for x in file.replace("good_cams.txt","")[-5:]])
    print(file,"seed", seed)
    
    # Calculate test size based on number of views
    n_views = len(good_views)
    if n_views <= 1:
        print(f"Warning: Only {n_views} view(s) found in {file}, skipping train/test split")
        continue
    
    # Ensure test_size is at least 1 but less than total views
    test_size = min(max(1, n_views - 6), n_views - 1)
    train_views, test_views = train_test_split(
        np.arange(n_views),
        test_size=test_size,
        random_state=seed
    )
    
    with open(file.replace("good_cams.txt",'train_val_v3.pkl'), 'wb+') as f:
        pickle.dump((train_views,test_views), f,protocol=2)
    scene = os.path.basename(os.path.dirname(file))
    result.append({'scene':scene,'n_good_views':len(good_views),'train':train_views,'test':test_views})
pd.DataFrame(result).sort_values('scene').to_csv("Num_good_views.csv",index=False)

# 新增：自动检查并补全cube.mat
# The original logic for scene_dirs should still work as 'files' now contains paths to 'good_cams.txt'
scene_dirs = [os.path.dirname(f) for f in files if os.path.exists(f)] # ensure f exists before os.path.dirname
for scene_dir in scene_dirs:
    mat_path = os.path.join(scene_dir, 'cube.mat')
    obj_path = os.path.join(scene_dir, 'cube.obj')
    if not os.path.isfile(mat_path):
        if os.path.isfile(obj_path):
            print(f"{scene_dir}: 未找到cube.mat，发现cube.obj，自动转换...")
            try:
                subprocess.run(['python', 'mat_obj_converter.py', 'obj2mat', obj_path, mat_path], check=True)
                print(f"{scene_dir}: 已生成cube.mat")
            except Exception as e:
                print(f"{scene_dir}: 转换cube.obj为cube.mat失败: {e}")
        else:
            print(f"{scene_dir}: 未找到cube.mat，也未找到cube.obj，请手动补充！")