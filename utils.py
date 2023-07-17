import difflib
import math
import os
import re
import paddle

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from scipy.stats import kruskal, chi2_contingency

def p_normalize(x, p=2):
    return x / (paddle.norm(x, p=p, axis=1, keepdim=True) + 1e-6)

def lifeline_analysis(df, title_g="brca"):
    '''
    :param df:
    生存分析画图，传入参数为df是一个DataFrame
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :param title_g: 图标题
    :return:
    '''
    n_groups = len(set(df["label"]))
    kmf = KaplanMeierFitter()
    plt.figure()
    for group in range(n_groups):
        idx = (df["label"] == group)
        kmf.fit(df['Survival'][idx], df['Death'][idx], label='class_' + str(group))

        ax = kmf.plot()
        plt.title(title_g)
        plt.xlabel("lifeline(days)")
        plt.ylabel("survival probability")
        treatment_median_confidence_interval_ = median_survival_times(kmf.confidence_interval_)
    plt.show()


# 富集分析
def clinical_enrichement(label,clinical):
    cnt = 0
    # age 连续 使用KW检验
    # print(label,clinical)
    stat, p_value_age = kruskal(np.array(clinical["age"]), np.array(label))
    if p_value_age < 0.05:
        cnt += 1
        # print("---age---")
    # 其余离散 卡方检验
    stat_names = ["gender","pathologic_T","pathologic_M","pathologic_N","pathologic_stage"]
    for stat_name in stat_names:
        if stat_name in clinical:
            c_table = pd.crosstab(clinical[stat_name],label,margins = True)
            stat, p_value_other, dof, expected = chi2_contingency(c_table)
            if p_value_other < 0.05:
                cnt += 1
                # print(f"---{stat_name}---")
    return cnt


def log_rank(df):
    '''
    :param df: 传入生存数据
    拥有字段：label（预测对标签） Survival（生存时间） Death（是否死亡）
    :return: res 包含了p log2p log10p
    '''
    res = dict()
    results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
    res['p'] = results.summary['p'].item()
    res['log10p'] = -math.log10(results.summary['p'].item())
    res['log2p'] = -math.log2(results.summary['p'].item())
    return res


def load_data_mean(path, cancer_type):
    path = os.path.join(path, cancer_type)
    exp = pd.read_csv(os.path.join(path, "exp"), sep="\t")
    methy = pd.read_csv(os.path.join(path, "methy"), sep="\t")
    mirna = pd.read_csv(os.path.join(path, "mirna"), sep="\t")
    survival = pd.read_csv(os.path.join(path, "survival"), sep="\t")

    return [exp, methy, mirna, survival]

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
    survival['age'] = -1 # 初始化年龄
    survival['gender'] = -1 # 初始化年龄
    if 'pathologic_T' in clinical.columns:
        survival['T'] = -1 # 初始化年龄
    if 'pathologic_M' in clinical.columns:
        survival['M'] = -1 # 初始化年龄
    if 'pathologic_N' in clinical.columns:
        survival['N'] = -1 # 初始化年龄
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['stage'] = -1 # 初始化年龄
    i = 0
    # 找对应的参数
    for name in survival['PatientID']:
        # print(name)
        flag = difflib.get_close_matches(name,list(clinical["sampleID"]),1,cutoff=0.6)
        if flag:
            idx = list(clinical["sampleID"]).index(flag[0])
            survival['age'][i] = clinical['age_at_initial_pathologic_diagnosis'][idx]
            survival['gender'][i] = clinical['gender'][idx]
            if 'pathologic_T' in clinical.columns:
                survival['T'][i] = clinical['pathologic_T'][idx]
            if 'pathologic_M' in clinical.columns:
                survival['M'][i] = clinical['pathologic_M'][idx]
            if 'pathologic_N' in clinical.columns:
                survival['N'][i] = clinical['pathologic_N'][idx]
            if 'tumor_stage.diagnoses' in clinical.columns:
                survival['stage'][i] = clinical['tumor_stage.diagnoses'][idx]
        else: print(name)
        i = i + 1
    return survival