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
def main():



    metric = ChineseMetricCalculator()

    hyp_list = 'CAD描述信息为,存在一个电梯,存在墙'
    ref_list = 'CAD描述信息为,存在墙,存在一个电梯'
    bleu_scores = metric.compute_bleu(ref_list, hyp_list)
    rouge_scores = metric.compute_rouge([ref_list], [hyp_list])
    bert_scores = metric.compute_bert_score([ref_list], [hyp_list])
    #bleu = calculate_average(bleu_scores)
    #rouge = calculate_average(rouge_scores)
    #bert = calculate_average(bert_scores)
    print(f"BLEU-4的分数是{bleu_scores}")
    print(f"ROUGE-L的分数是{rouge_scores}")
    print(f"BERTScore的分数是{bert_scores}")
if __name__ == "__main__":
    main()