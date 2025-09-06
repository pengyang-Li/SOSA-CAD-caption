import json
SVG_CATEGORIES = [  # categories
        # 1-6 doors
        {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "单扇门"},
        {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "双扇门"},
        {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "推拉门"},
        {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "折叠门"},
        {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "旋转门"},
        {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "卷帘门"},
        # 7-10 window
        {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "窗"},
        {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "凸窗"},
        {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "盲窗"},
        {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "开口"},
        # 11-27: furniture
        {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "沙发"},
        {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "床"},
        {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "椅子"},
        {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "桌子"},
        {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "电视柜"},
        {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "衣柜"},
        {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "储物柜"},
        {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "煤气炉"},
        {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "水槽"},
        {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "冰箱"},
        {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "空调装置"},
        {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "浴室"},
        {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "浴缸"},
        {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "洗衣机"},
        {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "蹲式厕所"},
        {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "小便池"},
        {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "坐便器"},
        # 28:stairs
        {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "楼梯"},
        # 29-30: equipment
        {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "电梯"},
        {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "自动扶梯"},

        # 31-35: uncountable
        {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "排椅子"},
        {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "停车位"},
        {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "墙"},
        {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "幕墙"},
        {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "栏杆"},
        {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
    ]
def find_instanceIndex(arr):
    """
        查找各个实例由哪些原语索引组成
    """
    instance_index = {}
    instance_index["num"] = max(arr)
    for index, value in enumerate(arr):
        if value > 0:
            if value not in instance_index:
                instance_index[value] = []
            instance_index[value].append(index)
    return instance_index

def find_semanticIndex(arr):
    """
        查找各个实例由哪些原语索引组成
    """
    semantic_index = {}
    for index, value in enumerate(arr):
        if value >=30 and value <=34:
            if value not in semantic_index:
                semantic_index[value] = []
            semantic_index[value].append(index)
    return semantic_index

def text_caption(json_file):
    """
    函数描述： 生成CAD图纸文本描述
    """
    data = json.load(open(json_file))
    instance_boxes = data["boxes"]

    # 输出各个可数实例的类别信息和位置信息
    combined_text = ''
    for i in instance_boxes:
        x = (i[0] + i[2]) / 2
        y = (i[1] + i[3]) / 2
        obj = i[4]
        name = SVG_CATEGORIES[obj]["name"]
        text = f"在（{x}，{y}）位置上有一个{name}"
        combined_text = ', '.join([combined_text, text])


    # 输出各个不可数原语的类别信息和位置信息
    semanticIds = data["semanticIds"]
    id_to_text = {
        31: "存在排椅子",
        32: "存在停车位",
        33: "存在墙",
        34: "存在幕墙",
        35: "存在栏杆"
    }
    # 遍历字典的键，检查它们是否在semanticIds中
    for id_num, text in id_to_text.items():
        if id_num-1 in semanticIds:
            combined_text = ', '.join([combined_text, text])
    print(f"有一张长100，宽100的cad图纸{combined_text}")

def dict_caption(json_file):
    """
    函数描述： 生成CAD图纸序列化描述
    """
    data = json.load(open(json_file))
    instance_boxes = data["boxes"]
    combined_text = {}
    for i in instance_boxes:
        x = round((i[0] + i[2]) / 2,2)
        y = round((i[1] + i[3]) / 2,2)
        obj = i[4]
        name = SVG_CATEGORIES[obj]["name"]
        if obj in {11, 16, 13, 28, 21, 15}:
            l = round(abs(i[0] - i[2]))
            w = round(abs(i[1] - i[3]))
            if name in combined_text:
                combined_text[name].append({'x': x, 'y': y, '长': l, '宽': w})  # 或其他你想要添加的元素
            else:
                combined_text[name] = [{'x': x, 'y': y, '长': l, '宽': w}]
        else:
            if name in combined_text:
                combined_text[name].append({'x': x, 'y': y})  # 或其他你想要添加的元素
            else:
                combined_text[name] = [{'x': x, 'y': y}]
    semanticIds = data["semanticIds"]
    id_to_text = {
        31: "排椅子",
        32: "停车位",
        33: "墙",
        34: "幕墙",
        35: "栏杆"
    }
    for id_num, name in id_to_text.items():
        if id_num-1 in semanticIds:
            if name in combined_text:
                combined_text[name].append()  # 或其他你想要添加的元素
            else:
                combined_text[name] = []
    print(combined_text)

def main():
    json_file = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset/ceshi/jsons/1332-0050.json"
    # text_caption(json_file)
    dict_caption(json_file)
if __name__ == "__main__":
    main()
    print("-------------------------break----------------------")