import os
import sys
from PIL import Image

def convert_and_rename_images(target_dir):
    list_path = os.path.join(target_dir, "bundle.out.list.txt")
    if not os.path.exists(list_path):
        print(f"未找到 {list_path}")
        return

    with open(list_path, "r") as f:
        png_files = [line.strip() for line in f if line.strip()]

    for idx, png_name in enumerate(png_files, start=1):
        png_path = os.path.join(target_dir, png_name)
        jpg_name = f"view{idx}.jpg"
        jpg_path = os.path.join(target_dir, jpg_name)

        if not os.path.exists(png_path):
            print(f"未找到图片: {png_path}，跳过")
            continue

        try:
            with Image.open(png_path) as im:
                rgb_im = im.convert("RGB")
                rgb_im.save(jpg_path, "JPEG")
            print(f"{png_name} -> {jpg_name}")
            os.remove(png_path)
        except Exception as e:
            print(f"转换 {png_name} 时出错: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("用法: python convert_png_to_jpg.py <目标目录>")
    else:
        convert_and_rename_images(sys.argv[1])