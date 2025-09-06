import json
import numpy as np
import torch
import re
import torch.nn as nn
import yaml
from munch import Munch
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import argparse
from create_data.caption_tool import *
import multiprocessing as mp
import os
import os.path as osp
import time
from functools import partial
from svgnet.data import build_dataloader, build_dataset
from svgnet.evaluation import PointWiseEval,InstanceEval
from svgnet.model.svgnet import SVGNet as svgnet
from svgnet.util  import get_root_logger, init_dist, load_checkpoint

def get_args():
    parser = argparse.ArgumentParser("svgnet")
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("checkpoint", type=str, help="path to checkpoint")
    # 是否使用同步批量归一化（sync_bn）
    parser.add_argument("--sync_bn", action="store_true", help="run with sync_bn")
    # 是否进行分布式训练
    parser.add_argument("--dist", action="store_true", help="run with distributed parallel")
    # 它是一个整数，用于设置随机种子，默认值为 2000
    parser.add_argument("--seed", type=int, default=2000)
    # 它是一个字符串，用于指定输出结果的目录
    parser.add_argument("--out", type=str, help="directory for output results")
    parser.add_argument("--save_lite", action="store_true")
    args = parser.parse_args()
    return args

def main():
    all_captions = {}
    args = get_args()
    cfg_txt = open(args.config, "r").read()     # 读取配置文件内容
    cfg = Munch.fromDict(yaml.safe_load(cfg_txt))  # 使用 Munch 库将 YAML 配置转换为 Python 对象，使其可以通过点号访问
    if args.dist:   # 如果需要分布式训练，则初始化分布式环境
        init_dist()
    logger = get_root_logger()  # 获取日志记录器

    model = svgnet(cfg.model).cuda()  # 实例化 svgnet 模型，并转移到 GPU 上
    if args.sync_bn:    # 如果指定了使用同步批归一化（sync_bn）
            # 将模型中的普通 BatchNorm 替换为 SyncBatchNorm
            nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.dist:   # 如果需要分布式训练
        # 将模型包装为 DistributedDataParallel 以支持分布式训练
        model = DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])
    gpu_num = dist.get_world_size()  # 获取当前分布式训练的世界大小（即总的 GPU 数量）
    logger.info(f"Load state dict from {args.checkpoint}")
    load_checkpoint(args.checkpoint, logger, model)  # 从检查点加载模型权重，并记录日志
    val_set = build_dataset(cfg.data.caption, logger)  # 根据配置文件中的测试数据设置构建数据集
    # 构建数据加载器，用于迭代数据集
    dataloader = build_dataloader(args, val_set, training=False, dist=args.dist, **cfg.dataloader.caption)

    time_arr = []   # 初始化一个空列表，用于记录每次迭代的时间

    # 初始化语义分割评估器
    #sem_point_eval = PointWiseEval(num_classes=cfg.model.semantic_classes,ignore_label=35,gpu_num=dist.get_world_size())
    # 初始化实例分割评估器
    #instance_eval = InstanceEval(num_classes=cfg.model.semantic_classes,ignore_label=35,gpu_num=dist.get_world_size())
    with torch.no_grad():   # 不计算梯度，开始评估模型
        model.eval()
        # 迭代数据加载器中的每个批次
        for i, batch in enumerate(dataloader):
            t1 = time.time()  # 记录当前批次开始的时间


            coord, feat, label, lengths, offset,json_file = batch
            batch = [coord, feat, label, lengths,offset]

            # print(json_file) # json_file = ['dataset/test/jsons/1322-0004.json']

            if i % 10 == 0:  # 每 10 个批次打印一次进度
                step = int(len(val_set) / gpu_num)  # 计算总的迭代步数
                logger.info(f"Infer  {i + 1}/{step}")  # 打印进度日志
            torch.cuda.empty_cache()  # 清理 GPU 缓存
            with torch.cuda.amp.autocast(enabled=cfg.fp16):  # 如果启用了混合精度训练（fp16）
                res = model(batch, return_loss=False)  # 将批次数据输入模型，并获取结果

            t2 = time.time()  # 记录当前批次结束的时间
            time_arr.append(t2 - t1)  # 计算并存储当前批次的处理时间


            # 生成描述。
            all_caption = text_caption(json_file, res['input'], res["instances"],res["targets"],res["lengths"])
            filename = os.path.splitext(os.path.basename(json_file[0]))[0]
            all_captions[filename] = all_caption
    # 将累积的字典写入一个 JSON 文件
    output_json_file = 'result.json'  # 可以根据需要更改文件名
    with open(output_json_file, 'w', encoding='utf-8') as f:
        json.dump(all_captions, f, ensure_ascii=False, indent=4)  # 使用 utf-8 编码和缩进格式化输出
    print(f"All captions have been written to {output_json_file}")

def text_caption(json_file, input, instances, target, lengths):
    """
    input : 存储所以原语的坐标，input['coords']是原语坐标。input['offsets']是每一张图片的分开索引
    instances: 一个列表，包含检测到的实例信息（如标签、得分和掩码）
    target: 一个字典，包含真实的目标信息（如标签和掩码）
    lengths: 一个Tensor，可能表示图像或特征图的尺寸或长度，用于计算面积

    """
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
    ignore_label = 35
    min_obj_score = 0.1
    IoU_thres = 0.5

    # 将传入的lengths进行对数变换并四舍五入到小数点后三位
    # 这样做可能是为了对数据进行某种形式的归一化或压缩
    lengths = np.round(np.log(1 + lengths.cpu().numpy()), 3)
    # 从target字典中提取labels和masks，并将它们从Tensor形式转换为numpy数组，然后将labels转换为列表
    tgt_labels = target["labels"].cpu().numpy().tolist()
    # print(target["labels"].cpu().numpy().shape)
    # 对tensor的第一个维度和第二个维度进行交换

    tgt_masks = target["masks"].transpose(0, 1).cpu().numpy()
    i = 0
    j = 0
    # for tgt_label, tgt_mask in zip(tgt_labels, tgt_masks):  # 遍历每一个真实的目标标签和对应的掩码
    #     if tgt_label == ignore_label: continue  # 如果该标签是预先设定的忽略标签，则跳过这个循环的剩余部分
    #     print(tgt_label)
    #     i += 1
    json_file = json_file[0]
    img_id = re.findall(r'\d+', json_file)  # img_id=['1332','0004']存储的是图像数字id
    data = json.load(open(json_file))
    args = np.array(data["args"]).reshape(-1, 8)   # 获取每个原语的坐标并进行归一化处理
    num = args.shape[0]  # 获取原语的数量
    max_num = max(num, 2048)
    coords = np.zeros((max_num, 2))
    coord_x = np.mean(args[:, 0::2], axis=1)  # 沿着列的方向计算x坐标的平均值（计算每一个原语的4个点的x坐标的平均值）
    coord_y = np.mean(args[:, 1::2], axis=1)
    # 将原始的2维坐标点转换成3维坐标点
    coords[:num, 0] = coord_x
    coords[:num, 1] = coord_y
    coords = coords.tolist()
    # print(coords)
    #coords = input['p_out'].cpu().numpy().tolist()
    offset = input['offset'].cpu().numpy().tolist()
    combined_text = {}
    for instance in instances:  # 遍历检测到的所有实例,筛选出不是被忽略标签且分数大于最小值的目标。
        src_label = instance["labels"]
        src_score = instance["scores"]
        if src_label == ignore_label: continue
        if src_score < min_obj_score: continue
        src_mask = instance["masks"]
        j += 1

        # 输出实例坐标位置
        # 使用 np.where 获取满足条件的索引（这里返回的是一个元组，其中第一个元素是我们需要的索引数组）
        indices = np.where(src_mask)[0]
        # 使用这些索引从 coord 中获取对应的元素
        coord = [coords[i] for i in indices]
        coord = np.array(coord)
        name = get_name_by_id(SVG_CATEGORIES, src_label + 1)

        if src_label<30:
            # 输出实例类别
            # print(f"class：{name}")
            # 计算坐标点x，y
            # 将coord坐标列表转换为NumPy数组，并重新塑形为(-1, 2)，即每行包含两个元素（x, y坐标）

            x1, y1 = np.min(coord[:, 0]), np.min(coord[:, 1])
            x2, y2 = np.max(coord[:, 0]), np.max(coord[:, 1])
            x = np.round((x1+x2)/2, 2)
            y = np.round((y1+y2)/2, 2)
            if src_label in {11, 16, 13,28,21,15}:
                l = np.round(x2 - x1,2)
                w = np.round(y2 - y1,2)
                if name in combined_text:
                    combined_text[name].append({'x': x, 'y': y, '长' : l, '宽': w})  # 或其他你想要添加的元素
                else:
                    combined_text[name] = [{'x': x, 'y': y, '长' : l, '宽': w}]
            else:
                if name in combined_text:
                    combined_text[name].append({'x': x, 'y': y})  # 或其他你想要添加的元素
                else:
                    combined_text[name] = [{'x': x, 'y': y}]

        else:
            if name in combined_text:
                continue
            else:
                combined_text[name] = "存在"

    # 分割路径以获取目录和文件名
    json_dir, json_filename = os.path.split(json_file)

    # 去除 JSON 文件的扩展名以获取基础文件名
    json_basename, _ = os.path.splitext(json_filename)

    # 替换目录中的 'jsons' 为 'svg'
    svg_dir = json_dir.replace('jsons', 'svg')

    # 构建 SVG 文件的完整路径
    svg_file = os.path.join(svg_dir, json_basename + '.svg')

    cad_bz_caption = dict_cad_bz_caption(svg_file)  # CAD标注信息
    final_caption = combination_caption(combined_text, cad_bz_caption)
    return final_caption
        #x1, y1 = np.min(coords[:, 0]), np.min(coords[:, 1])
        #x2, y2 = np.max(coords[:, 0]), np.max(coords[:, 1])
    # # 指定要写入的文件名
    # filename = './output.json'
    #
    # # 将字典写入JSON文件
    # with open(filename, 'w', encoding='utf-8') as json_file:
    #     json.dump(combined_text, json_file, ensure_ascii=False, indent=4)



    # print(f"CAD描述文本已写入 {filename}")
    # print(combined_text)
    # print(f'真实类别数{i}')
    # print(f'预测类别数{j}')




def get_name_by_id(categories, category_id):
    for category in categories:
        if category['id'] == category_id:
            return category['name']
    return None  # 如果没有找到匹配的id，则返回None



if __name__ == "__main__":
    main()
    print("-------------------------break----------------------")













