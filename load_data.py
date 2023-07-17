# 读取原始数据
import os
import re

import pandas as pd

def load_data(path):
    path = os.path.join("data", path)
    exp = pd.read_csv(os.path.join(path, "exp"), sep = " ")
    methy = pd.read_csv(os.path.join(path, "methy"), sep = " ")
    mirna = pd.read_csv(os.path.join(path, "mirna"), sep = " ")
    survival = pd.read_csv(os.path.join(path, "survival"), sep = "\t")
    survival = survival.dropna(axis=0)
    survival = survival.drop_duplicates() # BRCA 不建议使用
    if len(survival["PatientID"][survival.index[0]]) > len('tcga.16.1060'):
        name_list = list()
        survival["PatientID"] = [re.sub("-", ".", x) for x in survival["PatientID"].str.upper()]
        for token in survival["PatientID"]:
            if token[-2] != "0":
                survival.drop(survival[survival["PatientID"] == token].index, inplace = True)
                continue
            name_list.append(token)
            if token not in exp:
                exp[token] = exp.mean(axis=1)
            if token not in methy:
                methy[token] = methy.mean(axis=1)
            if token not in mirna:
                mirna[token] = mirna.mean(axis=1)
    else:
        survival["PatientID"] = [re.sub("-", ".", x) for x in survival["PatientID"].str.upper() + ".01"]
        for token in survival["PatientID"]:
            if token not in exp:
                exp[token] = exp.mean(axis=1)
            if token not in methy:
                methy[token] = methy.mean(axis=1)
            if token not in mirna:
                mirna[token] = mirna.mean(axis=1)
        name_list = survival["PatientID"]
    exp = exp[name_list]
    methy = methy[name_list]
    mirna = mirna[name_list]

    return [exp, methy, mirna, survival]


def load_data_mean(DATASET_PATH,path):
    path = os.path.join(DATASET_PATH, path)
    exp = pd.read_csv(os.path.join(path, "exp"), sep="\t")
    methy = pd.read_csv(os.path.join(path, "methy"), sep="\t")
    mirna = pd.read_csv(os.path.join(path, "mirna"), sep="\t")
    survival = pd.read_csv(os.path.join(path, "survival"), sep="\t")

    return [exp, methy, mirna, survival]