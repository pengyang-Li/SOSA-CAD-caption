
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..util import cuda_cast
from .pointtransformer import Model as PointT
#from .pointnet2 import Model as PointT
from .decoder import Decoder

import numpy as np

class SVGNet(nn.Module):
    def __init__(
        self,cfg,criterion=None):
        super().__init__()  # 调用父类nn.Module的初始化方法
        self.criterion = criterion  # 表示损失函数

        # NOTE backbone
        self.backbone = PointT(cfg)        #  用于从原始点云图像中提取多分辨率特征图（由于符号同类型可能尺寸方向都不相同），用于符号检测。
        self.decoder = Decoder(cfg,self.backbone.planes)    # Decoder类的实例化依赖于cfg配置对象和backbone输出的通道数
        self.num_classes = cfg.semantic_classes         # 语义类别数量
        self.test_object_score = 0.1                    # 测试类别分数阈值
        
    def train(self, mode=True):
        super().train(mode)
        
    def forward(self, batch,return_loss=True):
        coords,feats,semantic_labels,offsets,lengths = batch
        return self._forward(coords,feats,offsets,semantic_labels,lengths,return_loss=return_loss)
     
    def prepare_targets(self,semantic_labels,bg_ind=-1,bg_sem=35):
        """
            返回每一个实例的语义信息和该实例的位置掩码
            return [{
            "labels": [语义id],
            "masks": [每个语义id对应的索引位置掩码，......],

        }]
        """
        instance_ids = semantic_labels[:,1].cpu().numpy()
        semantic_ids = semantic_labels[:,0].cpu().numpy()
        
        keys = []    # 创建一个空的列表，用于存储唯一的(语义ID, 实例ID)对
        for sem_id,ins_id in zip(semantic_ids,
                             instance_ids):
            if (sem_id,ins_id) not in keys: # 如果这个(语义ID, 实例ID)对还没有在keys中出现过，就添加到keys中
                keys.append((sem_id,ins_id))
    
        cls_targets,mask_targets = [], []   # 初始化存储分类目标和掩码目标的列表
        svg_len = semantic_ids.shape[0]  # 获取semantic_ids的长度，即数据点的数量

        for (sem_id,ins_id) in keys:    # 遍历唯一的(语义ID, 实例ID)对
            if sem_id==35 and ins_id==-1: continue # background

            tensor_mask = torch.zeros(svg_len)  # 创建一个全为0的tensor，长度与原语数量相同，用于表示掩码
            ind1 = np.where(semantic_ids==sem_id)[0]    # 找到所有语义ID等于sem_id的数据点的索引
            ind2 = np.where(instance_ids==ins_id)[0]    # 找到所有实例ID等于ins_id的数据点的索引
            ind = list(set(ind1).intersection(ind2))    # 找出同时满足语义ID和实例ID条件的索引，即这两个索引集的交集
            tensor_mask[ind] = 1
            cls_targets.append(sem_id)
            mask_targets.append(tensor_mask.unsqueeze(1))

        # 如果cls_targets为空（即没有非背景的目标），则将其设置为背景ID的tensor
        cls_targets = torch.tensor(cls_targets) if cls_targets else torch.tensor([35])
        # 如果cls_targets为空（即没有非背景的目标），则将其设置为背景ID的tensor
        mask_targets = torch.cat(mask_targets,dim=1) if mask_targets else torch.zeros(svg_len,1)
        
        
        return [{
            "labels": cls_targets.to(semantic_labels.device),
            "masks": mask_targets.to(semantic_labels.device),

        }]

    @cuda_cast
    def _forward(
        self,
        coords,
        feats,
        offsets,
        semantic_labels,
        lengths,
        return_loss=True
    ):
        # 初始化一个字典，存储模型的中间输出和输入
        stage_list={'inputs': {'p_out':coords,"f_out":feats,"offset":offsets},"semantic_labels":semantic_labels[:,0]}
        targets = self.prepare_targets(semantic_labels)
        # 更新stage_list字典，添加目标标签
        stage_list.update({"tgt":targets})
        
        stage_list = self.backbone(stage_list)
        outputs = self.decoder(stage_list)

        model_outputs = {}      # 初始化一个字典，用于存储模型的最终输出
        if not self.training:   # 如果不是训练模式（self.training为False），则进行推理相关的计算
            # 对预测的logits和masks进行语义推理，得到语义分数
            semantic_scores=self.semantic_inference(outputs["pred_logits"],outputs["pred_masks"])
            # 对预测的logits和masks进行实例推理，得到实例信息
            instances = self.instance_inference(outputs["pred_logits"],outputs["pred_masks"])
            model_outputs.update(
                dict(
                semantic_scores=semantic_scores,
                ), 
            )   # 更新model_outputs字典，添加语义分数
       
            model_outputs.update(
                dict(
                semantic_labels=semantic_labels[:,0],
                    ), 
             )  #添加原始的语义标签（仅用于评估或可视化）
            model_outputs.update(
                dict(
                instances=instances,
                ),
            )

            model_outputs.update(
                dict(
                targets=targets[0],
                ),
            )   # 添加实例信息
            model_outputs.update(
                dict(
                lengths=lengths,
                ),
            )
            # 新添加的代码
            model_outputs.update(
                dict(
                    input = {'p_out':coords,"offset":offsets},
                ),
            )
         
        
        if not return_loss: # 如果不需要返回损失值（return_loss为False），则只返回模型的输出
            return model_outputs

        # NOTE cal loss
        # 以下是计算损失的部分
        # 使用定义的损失函数（self.criterion）计算损失
        losses = self.criterion(outputs,targets)
        loss_value,loss_dicts = self.parse_losses(losses)   # 解析损失，得到总的损失值和损失字典（可能包含多个子损失）
        
        
        return model_outputs,loss_value,loss_dicts  # 返回模型的输出、总损失值和损失字典

    
    def semantic_inference(self, mask_cls, mask_pred):

        # 对mask_cls进行softmax操作，沿着最后一个维度（通常是类别维度）进行，并去掉背景类别的预测（通常是最后一个通道）
        # softmax操作用于将输出转换为概率分布，确保所有类别的概率和为1
        # ... 表示省略的维度，比如batch_size和可能的空间维度（height, width）
        # Q,C 表示输入mask_cls的维度可能是(batch_size, 空间维度, 类别数)，C是类别数（不包括背景）
        mask_cls = F.softmax(mask_cls, dim=-1)[...,:-1] # Q,C
        mask_pred = mask_pred.sigmoid() # Q,G
        semseg = torch.einsum("bqc,bqg->bgc", mask_cls, mask_pred)
        return semseg[0]

    def instance_inference(self,mask_cls,mask_pred,overlap_threshold=0.8):
        # 定义一个实例推断函数，输入包括分类掩码（mask_cls）、预测掩码（mask_pred）和一个可选的重叠阈值（overlap_threshold
        mask_cls,mask_pred = mask_cls[0],mask_pred[0]
        # 对mask_cls进行softmax操作，并找到每个位置的最大值和对应的索引。这里的索引表示类别，值表示分数
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()

        # 过滤掉不属于任何类别的标签（即背景）和分数低于某个阈值的预测
        keep = labels.ne(self.num_classes) & (scores >= self.test_object_score)
        # 保留经过筛选的分数、类别、掩码和分类掩码（去掉最后一列，可能是背景类别）
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep][:, :-1]

        # 计算概率掩码，通过将分数与掩码相乘，得到每个像素属于某个类别的概率
        cur_prob_masks = cur_scores[..., None] * cur_masks
        current_segment_id = 0
        nline = cur_masks.shape[-1]

        results = []     # 初始化一个空列表，用于存储推断结果
        # take argmax
        try:
            cur_mask_ids = cur_prob_masks.argmax(0)
        except: 
            return results
        
        for k in range(cur_classes.shape[0]):

            pred_class = cur_classes[k].item()
            pred_score = cur_scores[k].item()
            mask_area = (cur_mask_ids == k).sum().item()
            original_area = (cur_masks[k] >= 0.5).sum().item()
            mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)
            if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                if mask_area / original_area < overlap_threshold:
                    continue
                current_segment_id += 1
                #print(pred_class, pred_score)
                results.append({
                    "masks": mask.cpu().numpy(),
                    "labels": pred_class,
                    "scores": pred_score
                })

        return results
     


    
    def parse_losses(self, losses):
        loss = sum(v for v in losses.values())
        losses["loss"] = loss
        for loss_name, loss_value in losses.items():
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            losses[loss_name] = loss_value.item()
        return loss, losses



