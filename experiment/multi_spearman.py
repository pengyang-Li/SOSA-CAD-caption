'''

多指标相关性验证
'''
import random
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from nltk.translate.bleu_score import sentence_bleu
import jieba
from rouge_chinese import Rouge
from bert_score import score as bert_score
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import eval_MDSAEM

random.seed(42)
np.random.seed(42)


# --------------------- 指标计算模块 ---------------------
class ChineseMetricCalculator:
    def __init__(self):
        self.rouge = Rouge()

    def compute_bleu(self, ref, hyp):
        """计算单个样本的BLEU-4分数"""
        ref_tokens = list(jieba.cut(ref))
        hyp_tokens = list(jieba.cut(hyp))
        return sentence_bleu([ref_tokens], hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))

    def compute_rouge(self, refs, hyps):
        """计算每个样本的ROUGE-L分数"""
        scores = []
        for ref, hyp in zip(refs, hyps):
            try:
                score = self.rouge.get_scores(hyp, ref)[0]["rouge-l"]["f"]
                scores.append(score)
            except:
                scores.append(0.0)
        return scores

    def compute_bert_score(self, refs, hyps):
        """计算每个样本的BERTScore"""
        _, _, f1 = bert_score(hyps, refs, lang="zh")
        return f1.numpy().tolist()

    def compute_md_saem(self, md_captions, md_results):
        """计算MD-SAEM分数（需根据实际实现调整）"""
        md_score = []
        for k in md_captions:
            md_caption=dict()
            md_result = dict()
            md_caption[k] = md_captions[k]
            md_result[k] = md_results[k]
            _, _, _,total_sum_score = eval_MDSAEM.run_MD_SAEM(md_caption,md_result)
            md_score.append(total_sum_score/100)
        return md_score


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def get_random_keys(data, num_keys):
    all_keys = list(data.keys())
    random_keys = random.sample(all_keys, num_keys)
    return random_keys


def generate_description(value):
    text_sum = ['CAD描述信息为']  # 使用列表来收集文本片段，便于后续处理
    non_cls = ['排椅子', '停车位', '墙', '幕墙', '栏杆']
    other_cls = ['其他标注信息', '空间标注信息']

    for name, details in value.items():
        text1 = ''
        if name in non_cls:
            text1 = f'存在{name}'
        elif name in other_cls:
            for ele in details:
                text2 = f'在({ele["x"]},{ele["y"]})处存在标注信息“{ele["txt"]}”'
                text1 += ',' if text1 else ''  # 只在非空时添加逗号
                text1 += text2
        else:
            for ele in details:
                text2 = f'在({ele["x"]},{ele["y"]})处有一个'
                if '长' in ele:
                    text2 += f'长{ele["长"]}宽{ele["宽"]}'
                if '标注信息' in ele:
                    if isinstance(ele["标注信息"], list):
                        text2 += '标注信息为:' + ','.join(ele["标注信息"]) + f'的{name}'
                    else:
                        text2 += '标注信息为:' + f'{ele["标注信息"]}' + f'的{name}'
                else:
                    text2 += f'{name}'
                text1 += ',' if text1 else ''  # 只在非空时添加逗号
                text1 += text2

        if text_sum:  # 如果text_sum非空，则在添加新文本前添加一个逗号（这里转换为字符串处理）
            text_sum.append(',')
        text_sum.append(text1.strip(','))  # 去除text1前后的多余逗号，再添加到列表中

    # 将列表转换为字符串并去除首尾可能的空格（虽然这里不会有空格，但这是个好习惯）
    return ''.join(text_sum).strip()


def main(json_file1, json_file2, num_keys=1000):
    # Step 1: Read JSON files
    data1 = read_json_file(json_file1)
    data2 = read_json_file(json_file2)
    del data2["info"]
    # Step 2: Get random keys from the first JSON file
    random_keys = get_random_keys(data1, num_keys)

    # Step 3: Traverse and print values for the random keys from the second JSON file
    refs = dict()
    hyps = dict()
    md_captions = dict()
    md_results = dict()
    for key in random_keys:
        value1 = data2.get(key, "Key not found in second JSON file")
        value2 = data1.get(key, "Key not found in second JSON file")
        value_1 = generate_description(value1)
        value_2 = generate_description(value2)
        refs[key] = value_1
        hyps[key] = value_2
        md_captions[key] = value1
        md_results[key] = value2
    data = [(refs[k], hyps[k]) for k in refs if k in hyps]
    ref_list = [d[0] for d in data]
    hyp_list = [d[1] for d in data]

    # 计算各指标得分
    metric = ChineseMetricCalculator()

    # 确保所有指标返回样本级分数列表
    bleu_scores = [metric.compute_bleu(ref, hyp) for ref, hyp in data]
    rouge_scores = metric.compute_rouge(ref_list, hyp_list)
    bert_scores = metric.compute_bert_score(ref_list, hyp_list)
    md_saem_scores = metric.compute_md_saem(md_captions, md_results)

    # 创建DataFrame
    df = pd.DataFrame({
        "BLEU-4": bleu_scores,
        "ROUGE-L": rouge_scores,
        "BERTScore": bert_scores,
        "MD-SAEM": md_saem_scores
    })

    # 计算相关系数矩阵
    corr_matrix = df.corr(method='spearman')

    # 5. 可视化
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title("Spearman Correlation Matrix")
    plt.tight_layout()
    plt.savefig("corr_matrix.png")
    plt.close()
    # 打印结果
    print("Spearman秩相关系数矩阵：")
    print(corr_matrix.round(2))

    # 显著性检验示例
    rho, p = spearmanr(df['MD-SAEM'], df['BLEU-4'])
    print(f"\nMD-SAEM与BLEU-4相关性检验：rho={rho:.3f}, p={p}")
    rho, p = spearmanr(df['MD-SAEM'], df['ROUGE-L'])
    print(f"\nMD-SAEM与ROUGE-L相关性检验：rho={rho:.3f}, p={p}")
    rho, p = spearmanr(df['MD-SAEM'], df['BERTScore'])
    print(f"\nMD-SAEM与BERTScore相关性检验：rho={rho:.3f}, p={p}")

if __name__ == "__main__":
    json_file1 = "/home/jupyter-lpy/project/target_detection/SymPoint_main/result.json"  # Contains keys
    json_file2 = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_FloorplanCAD/test/captions.json"  # Contains values (possibly for some of the keys in first.json)
    main(json_file1, json_file2)