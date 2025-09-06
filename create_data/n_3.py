import os
import re


"""
    更换语义id，使得与sympoint的标签一致
"""
# 定义要替换的映射关系
replacements = {
    '1': '88',
    '2': '99',
    '3': '1',
    '4': '2',
    '5': '3',
    '6': '4',
    '7': '5',
    '8': '6',
    '9': '7',
    '10': '8',
    '11': '9',
    '12': '10',
    '13': '11',
    '14': '12',
    '15': '13',
    '16': '14',
    '17': '15',
    '18': '16',
    '19': '17',
    '22': '18',
    '23': '19',
    '24': '22',
    '25': '23',
    '26': '24',
    '27': '25',
    '28': '26',
    '29': '27',
    '30': '28',
    '31': '29',
    '32': '30',
    '34': '31',
    '35': '32',
    '33': '35',
    '88': '33',
    '99': '34',
}

# 定义要处理的目录路径
directory_path = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_floorplanCAD/FloorPlanCAD/"

# 遍历目录中的所有文件
for filename in os.listdir(directory_path):
    if filename.endswith('.svg'):
        file_path = os.path.join(directory_path, filename)

        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as file:
            file_contents = file.read()

        # 使用正则表达式进行替换
        for old_value, new_value in replacements.items():
            pattern = r'semanticId="{}"'.format(old_value)
            new_pattern = 'semanticId="{}"'.format(new_value)
            file_contents = re.sub(pattern, new_pattern, file_contents)

        # 将修改后的内容写回到文件中
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(file_contents)

print("所有文件已处理完毕。")