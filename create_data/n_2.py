import os
import xml.etree.ElementTree as ET

"""
    更改svg文件中的属性名称，与sympoint的保持一致
"""
# 定义要修改的属性名称对
attribute_renames = {
    'instance-id': 'instanceId',
    'semantic-id': 'semanticId'
}

# 定义要处理的 SVG 文件所在的目录
svg_directory = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_floorplanCAD/FloorPlanCAD/"

# 遍历目录中的所有文件
for filename in os.listdir(svg_directory):
    if filename.endswith('.svg'):
        # 构建文件的完整路径
        file_path = os.path.join(svg_directory, filename)

        try:
            # 解析 SVG 文件
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 遍历所有元素，修改属性名称
            for elem in root.iter():
                for old_name, new_name in attribute_renames.items():
                    if old_name in elem.attrib:
                        # 将旧属性名称的值赋给新属性名称，并删除旧属性
                        elem.set(new_name, elem.attrib.pop(old_name))

            # 保存修改后的 SVG 文件（覆盖原文件）
            tree.write(file_path, xml_declaration=True, encoding='utf-8')
            # print(f"Successfully processed {filename}")

        except ET.ParseError:
            print(f"Error parsing {filename}: It may not be a valid SVG file.")
        except Exception as e:
            print(f"An unexpected error occurred while processing {filename}: {e}")

print("All SVG files have been processed.")