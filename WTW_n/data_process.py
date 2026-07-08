import os
import shutil
import numpy as np
from PIL import Image
import cv2
import torchvision.transforms as transforms  # 其实这版用不到 Resize，但保留也没关系

# ---------- 1. 自适应直方图均衡化（保持灰度图） ----------
def adaptive_histogram_equalization(image):
    """
    image: PIL Image (L)
    return: PIL Image after CLAHE
    """
    image_cv = np.array(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized_image = clahe.apply(image_cv)
    equalized_image_pil = Image.fromarray(equalized_image)
    return equalized_image_pil

# ---------- 2. 只做去标志 & 去方块，不裁剪、不缩放 ----------
def remove_marks_keep_size(image_path):
    """
    读入图像 → CLAHE → 把左上角标志和右下角方块擦掉 → 返回保持原尺寸的 PIL 图像
    """
    # 读图并转成灰度
    image = Image.open(image_path).convert('L')

    # CLAHE
    image_equalized = adaptive_histogram_equalization(image)
    img_array = np.array(image_equalized)

    # ================== 根据你原来的坐标进行遮挡 ==================
    # 左上角标志
    img_array[85:175, 41:140] = 0

    # 右下角（或左下角）的方块
    img_array[698:800, 1:150] = 0

    # 不再裁剪 img_array[10:, 10:]
    # 不再 Resize，保持原始尺寸

    img_processed = Image.fromarray(img_array)
    return img_processed

# ---------- 3. 找到所有叶子文件夹 ----------
def find_leaf_folders(root_dir):
    leaf_folders = []
    for dirpath, dirnames, _ in os.walk(root_dir):
        if not dirnames:  # 没有子目录就是叶子目录
            leaf_folders.append(dirpath)
    return leaf_folders

# ---------- 4. 删除所有旧的 processed 文件夹 ----------
def delete_processed_directories(root_dir):
    for dirpath, dirnames, _ in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            if dirname == 'processed':
                full_path = os.path.join(dirpath, dirname)
                print(f"删除旧目录: {full_path}")
                shutil.rmtree(full_path)

# ---------- 5. 批量处理图像（只去标志，保留尺寸） ----------
def process_images(root_dir):
    # 先删旧的 processed 目录
    delete_processed_directories(root_dir)

    leaf_folders = find_leaf_folders(root_dir)
    print("找到叶子文件夹:")
    for folder in leaf_folders:
        print(" ", folder)

    for folder in leaf_folders:
        parent_folder = os.path.dirname(folder)
        output_folder = os.path.join(parent_folder, 'processed')
        os.makedirs(output_folder, exist_ok=True)

        for filename in os.listdir(folder):
            if filename.lower().endswith(('.tif', '.png', '.jpg', '.jpeg', '.bmp')):
                image_path = os.path.join(folder, filename)
                output_path = os.path.join(output_folder, filename)

                img_processed = remove_marks_keep_size(image_path)
                img_processed.save(output_path)

                print(f"Processed (keep size) and saved to {output_path}")

# ---------- 6. 脚本入口 ----------
if __name__ == "__main__":
    # 这里改成你的数据根目录
    root_dir = '/Images_Input_yuantu'
    process_images(root_dir)
    print("任务完成！")
