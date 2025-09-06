import re
import math
import unicodedata
import xml.etree.ElementTree as ET

def contains_chinese(text):
    """
        判断是否含有中文字符
    """
    try:
        for char in text:
            if 'CJK' in unicodedata.name(char):
                return True
    except ValueError:
        # 可以记录日志或进行其他错误处理
        print(text)
    return False

def is_last_char_c_in_sequence(s):
    """
    判断是否属于窗户的标注信息
    """

    # 使用正则表达式查找连续的字母序列
    matches = re.findall(r'[a-zA-Z]+', s)

    # 遍历每个匹配的字母序列
    for match in matches:
        # 如果序列的最后一个字符是 'c'
        if match[-1] == 'C':
            # 检查这个序列后面是否不是字母字符（包括字符串末尾的情况）
            next_char_index = s.index(match) + len(match)
            if next_char_index >= len(s) or not is_latin_alpha(s[next_char_index]):
                return True

    # 如果没有找到符合条件的序列，返回False
    return False


def is_latin_alpha(char):
    """
    判断一个字符是否是拉丁字母（a-z或A-Z）。
    """
    return 'a' <= char <= 'z' or 'A' <= char <= 'Z'

def is_last_char_m_in_sequence(s):
    """
    判断是否属于门的标注信息，要求M后面不能紧跟加号，且M是字母序列的最后一个字符。
    """
    # 使用正则表达式查找连续的字母序列
    matches = re.findall(r'[a-zA-Z]+', s)

    # 遍历每个匹配的字母序列
    for match in matches:
        # 如果序列的最后一个字符是 'M'
        if match.endswith('M'):
            # 计算M后面的字符的索引位置
            next_char_index = s.find(match) + len(match)

            # 检查M后面是否紧跟加号或者是否还有字母字符
            if next_char_index >= len(s) or (
                    next_char_index < len(s) and not is_latin_alpha(s[next_char_index]) and s[next_char_index] != '+'):
                # 如果M后面是非字母字符且不是加号，或者M是字符串的最后一个字符，则返回True
                return True

    # 如果没有找到符合条件的序列，返回False
    return False




def contains_decimal_point(s):
    return bool(re.search(r'\.', s))


def contains_dt_or_elevator(s):
    """判断字符串中是否包含'DT'或者'电梯'"""
    return 'DT' in s or '电梯' in s or 'KG' in s


def is_point_in_elevator(elevators, a, b):
    """
    计算数据是否在电梯内部
    """
    for index, elevator in enumerate(elevators):
        x_center, y_center = float(elevator['x']), float(elevator['y'])
        length_half = float(elevator['长']) / 2
        width_half = float(elevator['宽']) / 2

        # 计算矩形的四个角点
        x1 = x_center - length_half
        y1 = y_center - width_half
        x2 = x_center + length_half
        y2 = y_center + width_half

        # 检查点是否在矩形内
        if x1 <= a <= x2 and y1 <= b <= y2:
            return index
    return -1


def is_point_near_elevator(elevators, a, b):
    """
    判断标注信息属于哪个电梯附近
    """
    nearest_index = None
    min_distance = float('inf')
    for index, elevator in enumerate(elevators):
        x_center, y_center = float(elevator['x']), float(elevator['y'])
        distance = math.sqrt((x_center - a) ** 2 + (y_center - b) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_index = index
    return nearest_index


def is_point_near_win_or_door(kinds, a, b):
    """
    判断标注信息属于哪个门或窗户
    """
    nearest_index = None
    min_distance = float('inf')
    for index, kind in enumerate(kinds):
        x_center, y_center = float(kind['x']), float(kind['y'])
        distance = math.sqrt((x_center - a) ** 2 + (y_center - b) ** 2)
        if distance < min_distance:
            min_distance = distance
            nearest_index = index
    return nearest_index, min_distance


def dict_cad_bz_caption(svg_file):
    """
        CAD图纸标注信息序列化
    """
    tree = ET.parse(svg_file)
    root = tree.getroot()
    ns = root.tag[:-3]
    ns1 = '{http://www.inkscape.org/namespaces/inkscape}'
    CAD_text = {}
    for g in root.iter(ns + 'g'):
        label = g.attrib[ns1 + 'label']  # 存储标签名
        for text in g.iter(ns + 'text'):
            txt = text.text.strip()
            x = str(round(float(text.attrib['x']), 2))
            y = str(round(float(text.attrib['y']), 2))
            color = text.attrib['fill']
            if label in CAD_text:
                CAD_text[label].append({'x': x, 'y': y, 'txt': txt})
            else:
                CAD_text[label] = [{'x': x, 'y': y, 'txt': txt}]
    return CAD_text


def combination_caption(caption_text,cad_text):
    for label, datas in cad_text.items():
        flag = True  # 标志位，判断是否全是数字
        for data in datas:
            if not data['txt'].isdigit():  # 判断是否全是数字
                flag = False
                break
        if flag:
            continue

        # for data in datas:
        #       print(data['txt'])
        #   print('------------')

        win_flag = False  # 判断这一图层是否存在窗户的标注信息
        door_flag = False  # 判断这一图层是否存在门的标注信息
        area_flag = False  # 判断这一层是否存在整体空间大小的标注信息
        dt_flag = False  # 判断这一图层是否存在电梯的标注信息
        for data in datas:
            if not win_flag:
                win_flag = is_last_char_c_in_sequence(data['txt'])
            if not door_flag:
                door_flag = is_last_char_m_in_sequence(data['txt'])
            if not area_flag:
                area_flag = contains_decimal_point(data['txt'])
            if not dt_flag:
                dt_flag = contains_dt_or_elevator(data['txt'])
        data_ = datas

        if win_flag or door_flag:
            # 将数据分为窗的数据、门的数据和其他数据，然后将其他数据与窗的数据进行匹配，以及与门的数据进行匹配
            win_data = []
            door_data = []
            other_data = []
            for data in data_:
                if is_last_char_c_in_sequence(data['txt']):
                    win_data.append(data)
                elif is_last_char_m_in_sequence(data['txt']):
                    door_data.append(data)
                else:
                    other_data.append(data)

            for i in range(len(win_data)):
                index = []
                for j, other_ in enumerate(other_data):
                    # 提取 win_data[i] 和 other_ 的 x 和 y 值
                    win_x, win_y = float(win_data[i]['x']), float(win_data[i]['y'])
                    other_x, other_y = float(other_['x']), float(other_['y'])
                    if (win_x == other_x and abs(win_y - other_y) < 13) or (
                            win_y == other_y and abs(win_x - other_x) < 13):
                        index.append(j)
                for w in index:
                    win_data[i]['x'] = round((float(win_data[i]['x']) + float(other_data[w]['x'])) / 2, 2)
                    win_data[i]['y'] = round((float(win_data[i]['y']) + float(other_data[w]['y'])) / 2, 2)
                    win_data[i]['txt'] = f"{win_data[i]['txt']}{other_data[w]['txt']}"

            for i in range(len(door_data)):
                index = []
                for j, other_ in enumerate(other_data):
                    # 提取 door_data[i] 和 other_ 的 x 和 y 值
                    door_x, door_y = float(door_data[i]['x']), float(door_data[i]['y'])
                    other_x, other_y = float(other_['x']), float(other_['y'])
                    if (door_x == other_x and abs(door_y - other_y) < 13) or (
                            door_y == other_y and abs(door_x - other_x) < 13):
                        index.append(j)
                for w in index:
                    door_data[i]['x'] = round((float(door_data[i]['x']) + float(other_data[w]['x'])) / 2, 2)
                    door_data[i]['y'] = round((float(door_data[i]['y']) + float(other_data[w]['y'])) / 2, 2)
                    door_data[i]['txt'] = f"{door_data[i]['txt']}{other_data[w]['txt']}"

            # print(win_data)
            # print(door_data)
            # print(other_data)
            # 将匹配后的数据与各类窗户进行匹配添加纸原数据中
            wins = ['窗', '凸窗', '盲窗']
            doors = ['单扇门', '双扇门', '推拉门', '折叠门', '旋转门', '卷帘门']
            for win_single_data in win_data:
                kind = None
                ind = None
                min_dis = float('inf')
                for win in wins:
                    x = float(win_single_data['x'])
                    y = float(win_single_data['y'])
                    if win in caption_text:
                        i = None
                        dis = None
                        i, dis = is_point_near_win_or_door(caption_text[win], x, y)
                        if dis < min_dis:
                            kind = win
                            ind = i
                            min_dis = dis
                if kind and min_dis < 18:
                    caption_text[kind][ind]['标注信息'] = win_single_data['txt']

            for door_single_data in door_data:
                kind = None
                ind = None
                min_dis = float('inf')
                for door in doors:
                    x = float(door_single_data['x'])
                    y = float(door_single_data['y'])
                    if door in caption_text:
                        i = None
                        dis = None
                        i, dis = is_point_near_win_or_door(caption_text[door], x, y)
                        if dis < min_dis:
                            kind = door
                            ind = i
                            min_dis = dis
                if kind and min_dis < 18:
                    caption_text[kind][ind]['标注信息'] = door_single_data['txt']

        if dt_flag:
            """
            处理关于电梯的标注信息
            """
            i = 0
            while i < len(data_):
                if '电梯' not in caption_text:
                    break
                x = float(data_[i]['x'])
                y = float(data_[i]['y'])
                elevators = caption_text['电梯']
                index = is_point_in_elevator(elevators, x, y)  # 获取需要插入的电梯id
                if index != -1:
                    if '标注信息' in caption_text['电梯'][index]:
                        caption_text['电梯'][index]['标注信息'].append(data_[i]['txt'])
                    else:
                        caption_text['电梯'][index]['标注信息'] = [data_[i]['txt']]
                    data_.remove(data_[i])
                elif contains_dt_or_elevator(data_[i]['txt']):
                    index = is_point_near_elevator(elevators, x, y)  # 获取需要插入的电梯id
                    if '标注信息' in caption_text['电梯'][index]:
                        caption_text['电梯'][index]['标注信息'].append(data_[i]['txt'])
                    else:
                        caption_text['电梯'][index]['标注信息'] = [data_[i]['txt']]
                    data_.remove(data_[i])
                else:
                    i += 1
        if area_flag:
            # 将包含小数点的数据和不包含小数点的数据划分开
            point_data = []
            other_data = []
            for data in data_:
                if contains_decimal_point(data['txt']):
                    point_data.append(data)
                else:
                    other_data.append(data)

            # 遍历小数点的数据，查找与之匹配的数据，将匹配后的数据进行存储。

            matched_data = []  # 初始化存储匹配成功数据的列表
            for point in point_data:
                min_distance = float('inf')
                closest_other_point = None
                for other_point in other_data:
                    distance = math.sqrt((float(point['x']) - float(other_point['x'])) ** 2 + (
                                float(point['y']) - float(other_point['y'])) ** 2)


                    # 如果找到更近的点且距离小于 10
                    if distance < min_distance and distance < 10:
                        min_distance = distance
                        closest_other_point = other_point
                if closest_other_point:
                    matched_x = round((float(point['x']) + float(closest_other_point['x'])) / 2, 2)
                    matched_y = round((float(point['y']) + float(closest_other_point['y'])) / 2, 2)
                    matched_txt = f"{closest_other_point['txt']}{point['txt']}"
                    matched_data.append({'x': matched_x, 'y': matched_y, 'txt': matched_txt})
                    other_data.remove(closest_other_point)

            for j in other_data:  # 将不匹配的数据进行存储。
                if contains_chinese(j['txt']):
                    matched_data.append({'x': j['x'], 'y': j['y'], 'txt': j['txt']})
            if matched_data:
                if '空间标注信息' in caption_text:
                    for _data in matched_data:
                        caption_text['空间标注信息'].append(_data)
                else:
                    caption_text['空间标注信息'] = matched_data
        if not win_flag and not door_flag and not dt_flag and not area_flag:
            for data in data_:
                if contains_chinese(data['txt']):
                    if '其他标注信息' in caption_text:
                        caption_text['其他标注信息'].append(data)
                    else:
                        caption_text['其他标注信息'] = [data]
    return caption_text
