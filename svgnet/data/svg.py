import numpy as np
import torch
from torch.utils.data import Dataset
import os.path as osp
from glob import glob
import json
import math
import random
from .aug_utils import *

SVG_CATEGORIES = [ # categories
    #1-6 doors
    {"color": [224, 62, 155], "isthing": 1, "id": 1, "name": "single door"},
    {"color": [157, 34, 101], "isthing": 1, "id": 2, "name": "double door"},
    {"color": [232, 116, 91], "isthing": 1, "id": 3, "name": "sliding door"},
    {"color": [101, 54, 72], "isthing": 1, "id": 4, "name": "folding door"},
    {"color": [172, 107, 133], "isthing": 1, "id": 5, "name": "revolving door"},
    {"color": [142, 76, 101], "isthing": 1, "id": 6, "name": "rolling door"},
    #7-10 window
    {"color": [96, 78, 245], "isthing": 1, "id": 7, "name": "window"},
    {"color": [26, 2, 219], "isthing": 1, "id": 8, "name": "bay window"},
    {"color": [63, 140, 221], "isthing": 1, "id": 9, "name": "blind window"},
    {"color": [233, 59, 217], "isthing": 1, "id": 10, "name": "opening symbol"},
    #11-27: furniture
    {"color": [122, 181, 145], "isthing": 1, "id": 11, "name": "sofa"},
    {"color": [94, 150, 113], "isthing": 1, "id": 12, "name": "bed"},
    {"color": [66, 107, 81], "isthing": 1, "id": 13, "name": "chair"},
    {"color": [123, 181, 114], "isthing": 1, "id": 14, "name": "table"},
    {"color": [94, 150, 83], "isthing": 1, "id": 15, "name": "TV cabinet"},
    {"color": [66, 107, 59], "isthing": 1, "id": 16, "name": "Wardrobe"},
    {"color": [145, 182, 112], "isthing": 1, "id": 17, "name": "cabinet"},
    {"color": [152, 147, 200], "isthing": 1, "id": 18, "name": "gas stove"},
    {"color": [113, 151, 82], "isthing": 1, "id": 19, "name": "sink"},
    {"color": [112, 103, 178], "isthing": 1, "id": 20, "name": "refrigerator"},
    {"color": [81, 107, 58], "isthing": 1, "id": 21, "name": "airconditioner"},
    {"color": [172, 183, 113], "isthing": 1, "id": 22, "name": "bath"},
    {"color": [141, 152, 83], "isthing": 1, "id": 23, "name": "bath tub"},
    {"color": [80, 72, 147], "isthing": 1, "id": 24, "name": "washing machine"},
    {"color": [100, 108, 59], "isthing": 1, "id": 25, "name": "squat toilet"},
    {"color": [182, 170, 112], "isthing": 1, "id": 26, "name": "urinal"},
    {"color": [238, 124, 162], "isthing": 1, "id": 27, "name": "toilet"},
    #28:stairs
    {"color": [247, 206, 75], "isthing": 1, "id": 28, "name": "stairs"},
    #29-30: equipment
    {"color": [237, 112, 45], "isthing": 1, "id": 29, "name": "elevator"},
    {"color": [233, 59, 46], "isthing": 1, "id": 30, "name": "escalator"},

    #31-35: uncountable
    {"color": [172, 107, 151], "isthing": 0, "id": 31, "name": "row chairs"},
    {"color": [102, 67, 62], "isthing": 0, "id": 32, "name": "parking spot"},
    {"color": [167, 92, 32], "isthing": 0, "id": 33, "name": "wall"},
    {"color": [121, 104, 178], "isthing": 0, "id": 34, "name": "curtain wall"},
    {"color": [64, 52, 105], "isthing": 0, "id": 35, "name": "railing"},
    {"color": [0, 0, 0], "isthing": 0, "id": 36, "name": "bg"},
]

class SVGDataset(Dataset):

    CLASSES = tuple([x["name"] for x in SVG_CATEGORIES])

    def __init__(self, data_root, split,data_norm,aug, repeat=1, logger=None):
        """
            # 参数包括：
            #   data_root: 数据根目录
            #   split: 数据集的分割类型（例如：训练集、验证集等）
            #   data_norm: 数据归一化方法或参数
            #   aug: 数据增强方法或参数
            #   repeat: 数据重复的次数（默认为1）
            #   logger: 日志记录器（默认为None）
        """
        self.split = split
        self.data_norm = data_norm
        self.aug = aug
        self.repeat = repeat
        self.data_list = glob(osp.join(data_root,"*.json")) # 返回一个包含匹配文件路径名的列表。
        logger.info(f"Load {split} dataset: {len(self.data_list)} svg") # 在程序中生成一条信息级别的日志消息，
        self.data_idx = np.arange(len(self.data_list)) # 生成一个等差数组，从0到len(self.data_list)-1
        
        self.instance_queues = []# 创建一个存储实例信息的队列

    def __len__(self):
        return len(self.data_list)*self.repeat
    
    @staticmethod
    def load(json_file,idx,min_points=2048):
        """
            定义load函数，用于加载JSON文件中的数据,将所有数据进行归一化处理

            json_file="dataset/test/jsons/1322-0004.json"
        """

        data = json.load(open(json_file))
        args = np.array(data["args"]).reshape(-1,8)/ 140 # 获取每个原语的坐标并进行归一化处理
        num = args.shape[0]     # 获取原语的数量
        max_num = max(num,min_points)
        
        coord = np.zeros((max_num,3))
        coord_x = np.mean(args[:,0::2],axis=1) # 沿着列的方向计算x坐标的平均值（计算每一个原语的4个点的x坐标的平均值）
        coord_y = np.mean(args[:,1::2],axis=1) 
        coord_z = np.zeros((num,)) # 创建一个长度为num的一维数组
        
        #coord_x = 2 * coord_x - 1
        #coord_y = 2 * coord_y - 1

        # 将原始的2维坐标点转换成3维坐标点
        coord[:num,0] = coord_x
        coord[:num,1] = coord_y
        coord[:num,2] = coord_z

        lengths = np.zeros(max_num)
        lengths[:num] = np.array(data["lengths"]) # 存储原语的长度
        
        feat = np.zeros((max_num,6))    # 存储原语的角度，长度和类别索引
        arc = np.arctan(coord_y/(coord_x + 1e-8)) / math.pi # 角度
        lens = np.array(data["lengths"]).clip(0,140) / 140  # 使用clip方法将数组中的值限制在0和140之间。

        # 这种操作通常用于创建独热编码（one-hot encoding），这是一种将类别变量转换为机器学习算法易于利用的格式的方法。
        # 表示按照data["commands"]这个索引列表进行单位矩阵的行的选择。（因为原语的类型只有4种）
        ctype = np.eye(4)[data["commands"]]

        feat[:num,0] = arc      # 角度
        feat[:num,1] = lens     # 长度
        feat[:num,2:] = ctype   # 原语类别索引

        # 初始化一个形状与coord[:, 0]相同的数组，所有元素为35，代表背景语义ID
        semanticIds = np.full_like(coord[:,0],35) # bg sem id = 35
        seg = np.array(data["semanticIds"])
        semanticIds[:num] = seg
        semanticIds = semanticIds.astype(np.int64)

        # 初始化一个形状与coord[:, 0]相同的数组，所有元素为-1，代表stuff的实例ID
        instanceIds = np.full_like(coord[:,0],-1) # stuff id = -1 存储不可计数的事物为-1
        ins = np.array(data["instanceIds"])
        valid_pos = ins != -1
        ins[valid_pos] += idx*min_points    # 对于有效的实例ID位置，加上idx与min_points的乘积，可能是为了区分不同的实例组
        
        instanceIds[:num] = ins
        instanceIds = instanceIds.astype(np.int64)
        # 将semanticIds和instanceIds拼接成一个二维数组，形成最终的标签数据
        label = np.concatenate([semanticIds[:,None],instanceIds[:,None]],axis=1)
        return coord, feat, label,lengths # 三维坐标点（coord），feat(存储原语的角度，长度和类别索引),label(语义和实例)
    
    def __getitem__(self, idx):
        """
            在创建对象时，在使用【】是会返回该索引下的coord（坐标），feat（特征），label（标签）和  lengths（长度）训练集没有，测试集有。
        """
        data_idx = self.data_idx[idx % len(self.data_idx)]
        # 计算实际的数据索引。这里使用了模运算（%），确保索引值在self.data_idx的范围内
        # 这样做可能是为了处理数据循环或数据增强的情况
        json_file = self.data_list[data_idx]




        # 加载的数据包括坐标(coord)、特征(feat)、标签(label)和长度(lengths)
        coord, feat, label,lengths = SVGDataset.load(json_file,idx)
        
        if self.split=="train":
            return self.transform_train(coord, feat, label)
        else:
            return self.transform_test(coord, feat, label,lengths,json_file)
    
    def transform_train(self,coord, feat, label):
        """
            定义训练数据的增强方法，接收坐标、特征和标签作为输入
        """
        # hflip
        # 如果启用了水平翻转且随机概率小于设定的增强概率
        if self.aug.hflip and np.random.rand() < self.aug.aug_prob:
            args = RandomHorizonFilp(coord[:,:2],width=1) # 因为所有的坐标点都是在0-1之间
            coord[:,:2] = args
        
        # vflip
        if self.aug.vflip and np.random.rand() < self.aug.aug_prob:
            args = RandomVerticalFilp(coord[:,:2],Hight=1)
            coord[:,:2] = args
            
        # rotate 旋转
        # 如果启用了旋转且随机概率小于设定的增强概率
        if self.aug.rotate.enable and np.random.rand() < self.aug.aug_prob:
            _min, _max = self.aug.rotate.angle
            angle = random.uniform(_min,_max)
            args = rotate_xy(coord[:,:2],width=1,height=1,angle=angle)
            coord[:,:2] = args
        
        if self.aug.rotate2 and np.random.rand() < self.aug.aug_prob:
            args = random_rotate(coord[:,:2],width=1,height=1)
            coord[:,:2] = args
        
        # random shift
        if self.aug.shift.enable and np.random.rand() < self.aug.aug_prob:
            _min, _max = self.aug.shift.scale
            scale = np.random.uniform(_min, _max,3)  # 在平移范围内随机生成一个平移向量
            scale[2] = 0    # 通常平移不涉及第三维（如深度或高度），所以设置为0
            coord += scale  # 对坐标进行平移
            
        # random scale  随机缩放
        if self.aug.scale.enable and np.random.rand() < self.aug.aug_prob: 
            _min, _max = self.aug.scale.ratio
            scale = np.random.uniform(_min, _max,1)
            coord *= scale
            feat[:,1] = feat[:,1] * scale


        mix_coord, mix_feat, mix_label = [], [], [] # 初始化用于存储混合后数据的列表
        mix_coord.append(coord)
        mix_feat.append(feat)
        mix_label.append(label)
        
        # random cutmix（一种数据增强方法：混合2张图片及标签）
        if self.aug.cutmix.enable and np.random.rand() < self.aug.aug_prob:
            
            unique_label = np.unique(label,axis=0)  # 查找行唯一的元素。
            # 遍历唯一的语义和实例标签组合
            for sem,ins in unique_label:
                if sem >=30: continue
                # 获取当前语义和实例标签对应的原语索引
                valid = np.logical_and(label[:,0]==sem,label[:,1]==ins)
                if len(self.instance_queues)<=self.aug.cutmix.queueK:   # 如果实例队列长度小于设定的队列大小
                    self.instance_queues.insert(0,{ # 将当前样本的坐标、特征和标签插入队列的开头
                        "coord":coord[valid],
                        "feat": feat[valid],
                        "label": label[valid]
                        })
                else:
                    self.instance_queues.pop()  # 如果队列已满，则弹出队列末尾的样本
            _min, _max = self.aug.cutmix.relative_shift # 获取 CutMix 相对于图像大小的随机平移范围
            rand_pos = np.random.uniform(_min, _max,3)  # 生成一个随机的平移向量
            rand_pos[2] = 0

            # 遍历实例队列中的每个实例，将每个实例插入到混合数据的列表中
            for instance in self.instance_queues:
                mix_coord.append(instance["coord"]+rand_pos) # random shift
                mix_feat.append(instance["feat"])
                mix_label.append(instance["label"])
        
        coord = np.concatenate(mix_coord,axis=0)    # mix_coord里存储的应该是2个列表
        feat = np.concatenate(mix_feat,axis=0)      # mix_feat里存储的应该是2个列表
        feat[:,0] = np.arctan(coord[:,1]/(coord[:,0] + 1e-8)) / math.pi     # feature should be change
        
        label = np.concatenate(mix_label,axis=0)
        
        # shuffle
        # 打乱坐标、特征和标签的顺序
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord, feat = coord[shuf_idx], feat[shuf_idx]
        if label is not None:   # 如果标签存在，则也打乱其顺序
            label = label[shuf_idx]
            
        # coord norm ----根据设定的归一化方式对坐标进行归一化
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)
        # 返回转换为张量的坐标、特征和标签
        return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label) , None

    def transform_test(self,coord, feat, label,lengths,json_file):
        
        # coord norm
        if self.data_norm == 'mean':
            coord -= np.mean(coord, 0)
        elif self.data_norm == 'min':
            coord -= np.min(coord, 0)
        return torch.FloatTensor(coord), torch.FloatTensor(feat), torch.LongTensor(label), torch.FloatTensor(lengths),json_file
        

    def collate_fn(self,batch):
        # batch可能是一个由多个元组组成的列表，每个元组包含四个元素：coord, feat, label, lengths
        coord, feat, label,lengths,json_file = list(zip(*batch))
        offset, count = [], 0       # offset列表用于在后续的模型处理中确定每个样本在合并后的数据中的起始位置。这在处理变长序列（如RNN中的序列数据）时尤为重要。
        # 遍历coord列表（实际上遍历的是batch中的每个样本的coord部分）
        for item in coord:
            count += item.shape[0]  # 累加当前样本的coord的数量（或长度）到count中
            offset.append(count)
        lengths = torch.cat(lengths) if lengths[0] is not None else None
        return torch.cat(coord), torch.cat(feat), torch.cat(label), torch.IntTensor(offset),lengths,json_file

