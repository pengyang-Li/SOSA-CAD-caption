from scipy.optimize import linear_sum_assignment
import numpy as np
import argparse
import json

def get_args():
    parser = argparse.ArgumentParser("evaluation")

    parser.add_argument("--captions", type=str, default='',help="original description")
    parser.add_argument("--results", type=str, default='',help="generate description")
    parser.add_argument("--save_file", type=str,default='', help="file save location")
    args = parser.parse_args()
    return args
def calculate_f1(tp, fp, fn):
    """计算F1分数"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0  # 计算精确率
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0     # 计算召回率
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0  # 计算F1


def match_instances(real, pred, is_other=False):
    """匈牙利算法匹配实例"""
    if not real or not pred or real=="存在" or pred=="存在":
        if pred=="存在":
            if real=="存在":
                return [], 0, 0, 0
            else:
                return [], 0, 0, len(real)

        elif pred!="存在":
            if real == "存在":
                return [], 0, len(pred), 0
            elif real != "存在":
                return [], 0, len(pred), len(real)
             # 如果真实或预测为空，返回空匹配

    # 构建代价矩阵
    cost_matrix = []
    for r in real:
        row = []
        for p in pred:
            if is_other:
                # 其他标注信息使用文本差异作为代价
                cost = 0 if r['txt'] == p['txt'] else 1
            else:
                # 空间标注使用坐标距离
                dx = float(r['x']) - float(p['x'])
                dy = float(r['y']) - float(p['y'])
                cost = (dx ** 2 + dy ** 2) ** 0.5
            row.append(cost)
        cost_matrix.append(row)

    # 执行匹配
    cost_matrix = np.array(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 统计匹配结果
    matched_pairs = []
    tp = 0
    for r, c in zip(row_ind, col_ind):
        if is_other:
            if cost_matrix[r, c] == 0:  # 文本匹配成功
                matched_pairs.append((r, c))
                tp += 1
        else:
            matched_pairs.append((r, c))
            tp += 1

    fp = len(pred) - tp  # 假阳性 = 预测总数 - 真阳性
    fn = len(real) - tp  # 假阴性 = 真实总数 - 真阳性
    return matched_pairs, tp, fp, fn

def has_non_empty_values(dictionary):
    for key, value in dictionary.items():
        if value and value != "存在":  # 这里使用了Python的“truthiness”检查，它会判断值是否为真（非空）
            return True
    return False


def run_MD_SAEM(captions,results):
    NONCOUNTABLE = ['排椅子', '停车位', '墙', '幕墙', '栏杆']
    OTHER_CATEGORYS = ['其他标注信息', '空间标注信息']  # 定义其他标注类别
    THRESHOLD = 15  # 坐标差阈值，超过此值认为不匹配
    WEIGHTS = [0.4, 0.4, 0.2]  # 三部分得分的权重（分类指标、坐标质量、特殊类别F1）

    part1_sum = []
    part2_sum = []
    part3_sum = []
    total_sum = []
    for key, caption in captions.items():
        if key != "info":
            # 从 results 字典中获取对应的结果
            result = results.get(key, "No result found")
            caption_keys = set(caption.keys())
            result_keys = list(set(result.keys()))
            caption_keys.discard('空间标注信息')
            caption_keys.discard('其他标注信息')
            CLS_CATEGORIES = list(caption_keys)  # 定义标注类别
            # ==================== 第一部分：类别标注分类指标 ====================
            cls_metrics = {'tp': 0, 'fp': 0, 'fn': 0}  # 初始化空间标注类别统计量
            part1_scores = []  # 存储每个类别的F1分数
            for cat in CLS_CATEGORIES:
                if cat in NONCOUNTABLE:
                    continue
                real = caption.get(cat, [])  # 获取真实实例
                pred = result.get(cat, [])  # 获取预测实例

                # 进行实例匹配
                _, tp, fp, fn = match_instances(real, pred)

                # 累计统计量
                cls_metrics['tp'] += tp
                cls_metrics['fp'] += fp
                cls_metrics['fn'] += fn

                # 计算当前类别F1
                category_f1 = calculate_f1(tp, fp, fn)
                part1_scores.append(category_f1)
            for cat in NONCOUNTABLE:
                if cat in CLS_CATEGORIES:
                    if cat in result_keys:
                        part1_scores.append(1)
                    else:
                        part1_scores.append(0)
            # 计算平均F1
            if not CLS_CATEGORIES:
                part1_score = 1
            else:
                part1_score = np.mean(part1_scores) if part1_scores else 0
            # ==================== 第二部分：坐标和标注质量评分 ====================
            part2_scores = []
            if has_non_empty_values({key: caption[key] for key in CLS_CATEGORIES if key in caption}):
                for cat in CLS_CATEGORIES:
                    real = caption.get(cat, [])
                    pred = result.get(cat, [])

                    # 获取匹配对
                    matched_pairs, _, _, _ = match_instances(real, pred)

                    for r_idx, p_idx in matched_pairs:
                        real_inst = real[r_idx]
                        pred_inst = pred[p_idx]

                        # 计算坐标差
                        dx = float(real_inst['x']) - float(pred_inst['x'])
                        dy = float(real_inst['y']) - float(pred_inst['y'])
                        distance = (dx ** 2 + dy ** 2) ** 0.5

                        # 计算得分
                        if distance > THRESHOLD:
                            score = 0
                        else:
                            # 坐标得分部分
                            coord_score = (1 - distance / THRESHOLD) * 0.5
                            # 标注正确性部分
                            if '标注信息' in real_inst and '标注信息' in pred_inst:
                                annot_correct = real_inst['标注信息'] == pred_inst['标注信息']
                                annot_score = 0.5 if annot_correct else 0
                                score = coord_score + annot_score
                            elif '标注信息' in real_inst or '标注信息' in pred_inst:
                                annot_score = 0
                                score = coord_score + annot_score
                            else:
                                score = coord_score * 2

                        part2_scores.append(score)

                part2_score = np.mean(part2_scores) if part2_scores else 0
            else:
                part2_score = 1
            # ==================== 第三部分：其他类别F1计算 ====================

            # 其他标注F1
            f1 = []
            for OTHER_CATEGORY in OTHER_CATEGORYS:
                real_other = caption.get(OTHER_CATEGORY, [])
                pred_other = result.get(OTHER_CATEGORY, [])
                _, tp_other, fp_other, fn_other = match_instances(real_other, pred_other, is_other=True)
                if not real_other and not pred_other:
                    f1.append(1)
                else:
                    f1.append(calculate_f1(tp_other, fp_other, fn_other))

            part3_score = np.mean(f1) if f1 else 0
            # ==================== 最终得分计算 ====================
            total_score = (
                    part1_score * WEIGHTS[0] +
                    part2_score * WEIGHTS[1] +
                    part3_score * WEIGHTS[2]
            )
            print(f"{key}:{total_score}")
            part1_sum.append(part1_score)
            part2_sum.append(part2_score)
            part3_sum.append(part3_score)
            total_sum.append(total_score)

    part1_sum_score = np.mean(part1_sum) if part1_sum else 0
    part2_sum_score = np.mean(part2_sum) if part2_sum else 0
    part3_sum_score = np.mean(part3_sum) if part3_sum else 0
    total_sum_score = np.mean(total_sum) if total_sum else 0

    return part1_sum_score,part2_sum_score,part3_sum_score,total_sum_score


if __name__ == "__main__":
    args = get_args()
    captions = json.load(open(args.captions))
    results = json.load(open(args.results))
    part1_sum_score, part2_sum_score, part3_sum_score, total_sum_score = run_MD_SAEM(captions,results)
    # ==================== 结果输出 ====================
    print(f"[类别标注分类指标]")
    print(f"平均F1: {part1_sum_score:.2%}")

    print(f"\n[类别的坐标和标注质量]")
    print(f"平均质量分: {part2_sum_score:.2%}")

    print(f"\n[特殊标注F1]")
    print(f"平均质量分: {part3_sum_score:.2%}")

    print(f"\n[最终得分]")
    print(f"加权总分: {total_sum_score:.2%}")
    print("-------------------------break----------------------")
