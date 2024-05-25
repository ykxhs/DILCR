import difflib
import math
import os
import re
import torch

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test
from lifelines.utils import median_survival_times
from matplotlib import pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.stats import kruskal, chi2_contingency
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, pair_confusion_matrix, accuracy_score


def p_normalize(x, p=2):
    return x / (torch.norm(x, p=p, dim=1, keepdim=True) + 1e-6)

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

def get_clinical(path,survival,cancer_type):
    clinical = pd.read_csv(f"{path}/{cancer_type}",sep="\t")
    if cancer_type == 'kirc':
        replace = {'gender.demographic': 'gender','submitter_id.samples': 'sampleID'}
        clinical = clinical.rename(columns=replace)  # 为某个 index 单独修改名称
        clinical["sampleID"] = [re.sub("A", "", x) for x in clinical["sampleID"].str.upper()]
    clinical["sampleID"] = [re.sub("-", ".", x) for x in clinical["sampleID"].str.upper()]
    survival['age'] = pd.NA # 初始化年龄
    survival['gender'] = pd.NA # 初始化年龄
    if 'pathologic_T' in clinical.columns:
        survival['T'] = pd.NA # 初始化年龄
    if 'pathologic_M' in clinical.columns:
        survival['M'] = pd.NA # 初始化年龄
    if 'pathologic_N' in clinical.columns:
        survival['N'] = pd.NA # 初始化年龄
    if 'tumor_stage.diagnoses' in clinical.columns:
        survival['stage'] = pd.NA # 初始化年龄
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
    return survival.dropna(axis=0, how='any')


def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    u = linear_sum_assignment(w.max() - w)
    ind = np.concatenate([u[0].reshape(u[0].shape[0], 1), u[1].reshape([u[0].shape[0], 1])], axis=1)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def get_rand_index_and_f_measure(labels_true, labels_pred, beta=1.):
     (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
     ri = (tp + tn) / (tp + tn + fp + fn)
     ari = 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
     p, r = tp / (tp + fp), tp / (tp + fn)
     f_beta = (1 + beta**2) * (p * r / ((beta ** 2) * p + r))
     return ri, ari, f_beta

def cluster_evaluate(label, pred):
    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    acc = cluster_acc(label, pred)
    pur = purity(label, pred)
    f_measure = get_rand_index_and_f_measure(label,pred)[2]
    return nmi, ari, acc, pur,f_measure