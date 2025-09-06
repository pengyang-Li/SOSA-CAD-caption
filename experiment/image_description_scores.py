from multi_spearman import ChineseMetricCalculator,read_json_file,generate_description
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


random.seed(42)
np.random.seed(42)

def calculate_average(numbers):
    if not numbers:  # 检查列表是否为空
        return 0  # 或者可以选择抛出异常，视需求而定
    total = sum(numbers)  # 计算列表中所有元素的总和
    count = len(numbers)  # 获取列表中的元素个数
    average = total / count  # 计算平均值
    return average
def main(json_file1, json_file2):
    data1 = read_json_file(json_file1)
    data2 = read_json_file(json_file2)
    del data2["info"]
    refs = dict()
    hyps = dict()
    for key,value in data1.items():
        hyps[key] = generate_description(value)
    for key,value in data2.items():
        refs[key] = generate_description(value)

    data = [(refs[k], hyps[k]) for k in refs if k in hyps]
    ref_list = [d[0] for d in data]
    hyp_list = [d[1] for d in data]
    # 计算各指标得分

    my_list_of_lists = [list(tuple_item) for tuple_item in data]

    with open('output11111111.txt', 'w', encoding='utf-8') as file:
        json.dump(my_list_of_lists, file, ensure_ascii=False, indent=4)

    print("内容已成功写入文件 output1111111.txt")



    metric = ChineseMetricCalculator()

    # 确保所有指标返回样本级分数列表
    bleu_scores = [metric.compute_bleu(ref, hyp) for ref, hyp in data]
    rouge_scores = metric.compute_rouge(ref_list, hyp_list)
    bert_scores = metric.compute_bert_score(ref_list, hyp_list)
    bleu = calculate_average(bleu_scores)
    rouge = calculate_average(rouge_scores)
    bert = calculate_average(bert_scores)
    print(f"BLEU-4的分数是{bleu}")
    print(f"ROUGE-L的分数是{rouge}")
    print(f"BERTScore的分数是{bert}")
if __name__ == "__main__":
    json_file1 = "/home/jupyter-lpy/project/target_detection/SymPoint_main/result.json"  # Contains keys
    json_file2 = "/home/jupyter-lpy/project/target_detection/SymPoint_main/dataset_FloorplanCAD/test/captions.json"  # Contains values (possibly for some of the keys in first.json)
    main(json_file1, json_file2)