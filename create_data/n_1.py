import os
import shutil
"""
根据sympoint的数据集要求（将相同文件名的svg划分训练集测试集等）将FloorPlanCAD数据集进行文件划分
"""

def move_matching_files(source_dir, target_dir_reference, svg_destination_dir, png_destination_dir):
    # 确保目标目录存在，如果不存在则创建
    if not os.path.exists(svg_destination_dir):
        os.makedirs(svg_destination_dir)
    if not os.path.exists(png_destination_dir):
        os.makedirs(png_destination_dir)

    # 获取文件夹B中的所有文件名（只获取文件名，不包含路径）
    reference_files = {os.path.splitext(f)[0] for f in os.listdir(target_dir_reference) if f.endswith('.svg')}

    # 遍历文件夹A中的所有文件
    for filename in os.listdir(source_dir):
        # 获取文件名（不包含路径）和扩展名
        base_filename, ext = os.path.splitext(filename)

        # 检查文件是否是SVG或PNG文件，并且文件名是否在文件夹B中存在
        if ext in ['.svg', '.png'] and base_filename in reference_files:
            # 构建文件的完整路径
            source_file_path = os.path.join(source_dir, filename)

            # 根据文件类型确定目标路径
            if ext == '.svg':
                destination_file_path = os.path.join(svg_destination_dir, filename)
            else:  # ext == '.png'
                destination_file_path = os.path.join(png_destination_dir, filename)

            # 移动文件
            shutil.move(source_file_path, destination_file_path)


# 文件夹路径
folder_A = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_floorplanCAD/FloorPlanCAD/"
folder_B = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_yuanshi/train/train/svg_gt/"
folder_C = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_floorplanCAD/train/svg/"  # 用于存放SVG文件
folder_D = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_floorplanCAD/train/png/"  # 用于存放PNG文件

# 调用函数进行文件整理
move_matching_files(folder_A, folder_B, folder_C, folder_D)