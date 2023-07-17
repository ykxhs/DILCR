import paddle
from paddle import nn
from paddle import optimizer
from paddle.io import TensorDataset,DataLoader
from paddle.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import sklearn.metrics as skm
from sklearn.cluster import KMeans,SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, accuracy_score
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score
# from sklearn.metrics.cluster import pair_confusion_matrix

from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import multivariate_logrank_test

import os
import zipfile
import math
import csv
import re
import argparse

import warnings

import utils
from load_data import load_data_mean
from model import DIMO_VAE, loss_funcation
from utils import p_normalize, lifeline_analysis, get_clinical, clinical_enrichement

warnings.filterwarnings("ignore")

paddle.seed(123456)
np.random.seed(123456)

#
DATASET_PATH = "D:/cyy/DIMO-VAE/data"

if __name__ == '__main__':
    # load data
    cancer_type = 'aml'
    conf = dict()
    conf['dataset'] = cancer_type
    exp, methy, mirna, survival = load_data_mean(DATASET_PATH,cancer_type)
    exp_df = paddle.to_tensor(exp.values[:, 1:].T, dtype=paddle.float32)
    methy_df = paddle.to_tensor(methy.values[:, 1:].T, dtype=paddle.float32)
    mirna_df = paddle.to_tensor(mirna.values[:, 1:].T, dtype=paddle.float32)
    full_data = [p_normalize(exp_df), p_normalize(methy_df), p_normalize(mirna_df)]

    # params
    conf = dict()
    conf['dataset'] = cancer_type
    conf['batch_size'] = 64
    conf['encoder_dim'] = [1024, 1024]
    conf['feature_dim'] = 512
    conf['peculiar_dim'] = 128
    conf['common_dim'] = 128
    conf['mu_logvar_dim'] = 10
    if (conf['dataset'] == "brca") | (conf['dataset'] == "skcm") | (conf['dataset'] == "lihc"):
        conf['cluster_num'] = 5
    if (conf['dataset'] == "coad") | (conf['dataset'] == "kirc"):
        conf['cluster_num'] = 4
    if (conf['dataset'] == "gbm") | (conf['dataset'] == "aml") | (conf['dataset'] == "ov") | (
            conf['dataset'] == "sarc") | (conf['dataset'] == "lusc"):
        conf['cluster_num'] = 3

    conf['cluster_var_dim'] = 3 * conf['common_dim']
    conf['up_and_down_dim'] = 512
    conf['view_num'] = 3
    conf['use_cuda'] = True
    conf['lr'] = 1e-5
    conf['min_lr'] = 1e-6
    conf['stop'] = 1e-6
    conf['kl_loss_lmda'] = 10

    lmda_list = dict()
    lmda_list['rec_lmda'] = 0.9
    lmda_list['KLD_lmda'] = 0.7
    lmda_list['I_loss_lmda'] = 0.7

    conf['pre_epochs'] = 2000
    conf['idec_epochs'] = 500
    conf['update_interval'] = 10
    eval_epoch = 50

    # train
    folder = "result/{}_result".format(conf['dataset'], conf['cluster_num'])
    if not os.path.exists(folder):
        os.makedirs(folder)

    in_dim = [exp_df.shape[1], methy_df.shape[1], mirna_df.shape[1]]
    model = DIMO_VAE(in_dim=in_dim, encoder_dim=conf['encoder_dim'], feature_dim=conf['feature_dim'],
                  common_dim=conf['common_dim'],
                  mu_logvar_dim=conf['mu_logvar_dim'], cluster_var_dim=conf['cluster_var_dim'],
                  up_and_down_dim=conf['up_and_down_dim'], cluster_num=conf['cluster_num'],
                  peculiar_dim=conf['peculiar_dim'], use_cuda=conf['use_cuda'], view_num=conf['view_num'])
    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.001)
    opt = paddle.optimizer.AdamW(learning_rate=conf['lr'], parameters=model.parameters(), grad_clip=clip)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(conf['lr'], T_max=conf['pre_epochs'], eta_min=conf['min_lr'])
    loss = loss_funcation()
    # print(model)
    print("pre-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))
    pbar = tqdm(range(conf['pre_epochs']), ncols=120)
    max_log = 0.0
    max_label, max_center = [], []
    label_list = []
    for epoch in pbar:
        # 抽取数据 训练batch
        sample_num = exp_df.shape[0]
        randidx = paddle.randperm(sample_num)
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [p_normalize(exp_df[idx]), p_normalize(methy_df[idx]), p_normalize(mirna_df[idx])]
            out_list, latent_dist = model(data_batch)

            l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                                latent_dist=latent_dist,
                                lmda_list=lmda_list, batch_size=conf['batch_size'])
            l.backward()
            opt.step()
            opt.clear_grad()
        # 评估模型
        if (epoch + 1) % eval_epoch == 0:
            with paddle.no_grad():
                model.eval()
                # 将整个数据放入求每一个fusion
                out_list, latent_dist = model(full_data)
                kmeans = KMeans(n_clusters=conf['cluster_num'], n_init=20, random_state=123456, init="k-means++").fit(
                    latent_dist['cluster_var'])
                pred = kmeans.labels_
                cluster_center = kmeans.cluster_centers_
                survival["label"] = np.array(pred)
                df = survival
                results = multivariate_logrank_test(df['Survival'], df['label'], df['Death'])
                p = results.summary['p'].item()
                log2p = -math.log2(p)
                log10p = -math.log10(p)

            # 遇到较大的 保存模型 保存标签
            if (log10p > max_log):
                max_log = log10p
                max_label = pred
                max_center = cluster_center
                paddle.save(model.state_dict(), "{}/{}_max_log.pdparams".format(folder, conf['dataset']))
                paddle.save(opt.state_dict(), "opt.pdopt")

        scheduler.step()
        pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                         rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                         KLD="{:3.4f}".format(loss_dict['KLD'].item()))

    # IDEC
    # 载入模型（预热的）
    in_dim = [exp_df.shape[1], methy_df.shape[1], mirna_df.shape[1]]
    label_model = DIMO_VAE(in_dim=in_dim, encoder_dim=conf['encoder_dim'], feature_dim=conf['feature_dim'],
                        common_dim=conf['common_dim'],
                        mu_logvar_dim=conf['mu_logvar_dim'], cluster_num=conf['cluster_num'],
                        peculiar_dim=conf['peculiar_dim'], use_cuda=conf['use_cuda'])
    load_model_name = "{}/{}_max_log.pdparams".format(folder, conf['dataset'])
    print("load_params:------{}".format(load_model_name))
    para_dict = paddle.load(load_model_name)
    label_model.load_dict(para_dict)
    out_list, latent_dist = label_model(full_data)

    # 初始化聚类中心
    # 如果不是从头训练 就需要重新初始化一下
    kmeans = KMeans(n_clusters=conf['cluster_num'], random_state=123456,init="k-means++").fit(latent_dist['cluster_var'])
    # label_model.cluster_layer = paddle.static.create_parameter([conf['cluster_num'], latent_dist['cluster_var'].shape[1]], dtype = "float32",default_initializer = nn.initializer.KaimingNormal())
    paddle.assign(paddle.to_tensor(kmeans.cluster_centers_, dtype=paddle.float32), label_model.cluster_layer)  # 初始化
    out_list, latent_dist = label_model(full_data)

    clip = paddle.nn.ClipGradByGlobalNorm(clip_norm=0.001)
    opt = paddle.optimizer.AdamW(learning_rate=conf['lr'], parameters=label_model.parameters(), grad_clip=clip)
    scheduler = paddle.optimizer.lr.CosineAnnealingDecay(conf['lr'], T_max=conf['idec_epochs'], eta_min=conf['min_lr'])
    loss = loss_funcation()

    y_pred = max_label
    y_pred_last = y_pred

    print("idec-----------------------train-dataset-: {} cluster_num-: {}".format(conf['dataset'], conf['cluster_num']))
    max_label_log = 0.0
    max_label_pred = []

    clinical_data = get_clinical(DATASET_PATH + "/clinical",survival, conf["dataset"])

    pbar = tqdm(range(conf['idec_epochs']), ncols=120)
    for epoch in pbar:
        # 初始化
        if epoch % conf['update_interval'] == 0:
            _, latent_dist = label_model(full_data)
            # print(label_model.cluster_layer)
            # update target distribution p
            tmp_q = latent_dist['q']
            weight = tmp_q ** 2 / tmp_q.sum(0)
            p = (weight.t() / weight.sum(1)).t()
            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            # 评估
            df = survival
            df["label"] = np.array(y_pred)
            res = utils.log_rank(df)
            cnt_NI = clinical_enrichement(y_pred, clinical_data)
            if res['log10p'] > max_label_log:
                max_label_log = res['log10p']
                max_label_pred = y_pred

            if epoch > 0 and delta_label < conf['stop']:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      conf['stop'])
                print('Reached tolerance threshold. Stopping training.')
                break

        # 抽取数据 训练batch
        sample_num = exp_df.shape[0]
        randidx = paddle.randperm(sample_num)
        for i in range(round(sample_num / conf['batch_size'])):
            idx = randidx[conf['batch_size'] * i:(conf['batch_size'] * (i + 1))]
            data_batch = [p_normalize(exp_df[idx]), p_normalize(methy_df[idx]), p_normalize(mirna_df[idx])]
            out_list, latent_dist = label_model(data_batch)
            tmp_q = latent_dist['q']
            # print(tmp_q.shape,p[idx].shape)
            kl_loss = F.kl_div(tmp_q.log(), p[idx])
            l, loss_dict = loss(view_num=conf['view_num'], data_batch=data_batch, out_list=out_list,
                                latent_dist=latent_dist,
                                lmda_list=lmda_list, batch_size=conf['batch_size'])
            # print(kl_loss.item())
            l += conf['kl_loss_lmda'] * kl_loss
            l.backward()
            opt.step()
            opt.clear_grad()

        scheduler.step()
        pbar.set_postfix(loss="{:3.4f}".format(loss_dict['loss'].item()),
                         rec_loss="{:3.4f}".format(loss_dict['rec_loss'].item()),
                         KLD="{:3.4f}".format(loss_dict['KLD'].item()),
                         KL_loss="{:3.4f}".format(kl_loss.item()),
                         )

    survival["label"] = np.array(max_label_pred)
    cnt_NI = clinical_enrichement(max_label, clinical_data)
    cnt = clinical_enrichement(max_label_pred, clinical_data)

    print("{}:    DIMO-VAE-NI:  {}/{:.1f}   DIMO-VAE-ALL:   {}/{:.1f}".format(conf['dataset'],cnt_NI,max_log,cnt,max_label_log))