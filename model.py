import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

def ContrastiveLoss(view_num=3, Mv_common_peculiar=[], temperature=0.95, batch_size=128):
    '''
        不同视图里面的同一样本的common作为正样本 peculiar全是负样本
    '''
    pos = 0.0  # 正样本得分
    neg = 0.0  # 负样本得分
    cnt = 0
    for view in range(view_num):
        common_i = F.normalize(Mv_common_peculiar[view][0], dim=1)
        for view_other in range(view + 1, view_num):
            common_j = F.normalize(Mv_common_peculiar[view_other][0], dim=1)
            pos += F.cosine_similarity(common_i, common_j, dim=1)
            cnt += 1
    pos /= (cnt * temperature)
    pos = torch.sum(torch.exp(pos)) / batch_size
    cnt = 0
    for view in range(view_num):
        common_view = F.normalize(Mv_common_peculiar[view][0], dim=1)
        for view_other in range(view_num):
            peculiar_view = F.normalize(Mv_common_peculiar[view_other][1], dim=1)
            neg += F.cosine_similarity(common_view, peculiar_view, dim=1)
            cnt += 1
    neg /= (cnt * temperature)
    neg = torch.sum(torch.exp(neg)) / batch_size
    return -torch.log(pos / (pos + neg))


class loss_funcation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, view_num, data_batch, out_list, latent_dist, lmda_list, batch_size):
        '''
            都是每个batch的值
        '''
        batch_size = data_batch[0].shape[1]
        Mv_common_peculiar = latent_dist['Mv_common_peculiar']
        mu, logvar = latent_dist['mu_logvar']  # 返回每一个视图的mu和logvar
        # 重构损失 KL散度 互信息 损失
        rec_loss, KLD, I_loss, loss = 0.0, 0.0, 0.0, 0.0
        # label 指导
        loss_dict = {}
        # 重构损失和VAE每个部分的KLD
        for i in range(view_num):
            rec_loss += torch.sum(torch.pow(data_batch[i] - out_list[i], 2))
            KLD += 0.5 * torch.sum(torch.exp(logvar[i]) + torch.pow(mu[i], 2) - 1. - logvar[i])
        rec_loss /= view_num
        KLD /= view_num
        # # 同一个视图的common 和 peculiar 应该尽量不相关
        # # 不同视图的common 应该尽量的相关
        I_loss = ContrastiveLoss(view_num,Mv_common_peculiar,batch_size = batch_size)
        loss = (lmda_list['rec_lmda'] * rec_loss + lmda_list['KLD_lmda'] * KLD + lmda_list['I_loss_lmda'] * I_loss)
        loss_dict['rec_loss'] = rec_loss
        loss_dict['KLD'] = KLD
        loss_dict['I_loss'] = I_loss
        loss_dict['loss'] = loss
        return loss, loss_dict


class DILCR(nn.Module):
    def __init__(self, in_dim=[], encoder_dim=[], feature_dim=512, peculiar_dim=512, common_dim=512,
                 mu_logvar_dim=10, cluster_var_dim=384, up_and_down_dim=512, cluster_num=5, view_num=3,
                 temperature=.67, device = 'cpu'):
        super(DILCR, self).__init__()
        self.device = device

        self.view_num = view_num
        self.mu_logvar_dim = mu_logvar_dim
        self.feature_dim = feature_dim
        self.common_dim = common_dim
        self.peculiar_dim = peculiar_dim
        self.in_dim = in_dim
        self.temperature = temperature
        self.fusion_dim = self.common_dim * 3
        self.cluster_num = cluster_num
        self.alpha = 1.0
        self.cluster_var_dim = cluster_var_dim
        self.up_and_down_dim = up_and_down_dim
        self.use_up_and_down = up_and_down_dim

        Mv_encoder_MLP = []
        Mv_feature_peculiar = []
        Mv_peculiar_to_mu_logvar = []
        Mv_feature_to_common = []
        # decoder
        Mv_decoder_MLP = []
        for i in range(self.view_num):
            # 输入到特征
            encoder_MLP = []
            # 输入维度不一致
            encoder_MLP += [
                nn.Linear(in_dim[i], encoder_dim[0]),
                nn.GELU()
            ]
            for j in range(len(encoder_dim) - 1):
                encoder_MLP += [
                    nn.Linear(encoder_dim[j], encoder_dim[j + 1]),
                    nn.GELU()
                ]
            # 最后一层MLP到特征
            encoder_MLP += [
                nn.Linear(encoder_dim[-1], feature_dim),
                nn.GELU()
            ]
            Mv_encoder_MLP.append(nn.Sequential(*encoder_MLP))
            # 特征到mu,logvar
            Mv_feature_peculiar.append(nn.Sequential(
                nn.Linear(self.feature_dim, self.peculiar_dim),
                nn.GELU()
            ))
            Mv_peculiar_to_mu_logvar.append(nn.Sequential(
                nn.Linear(self.peculiar_dim, self.mu_logvar_dim),
                nn.GELU()
            ))
            # 特征到common
            Mv_feature_to_common.append(nn.Sequential(
                nn.Linear(self.feature_dim, self.common_dim),
                nn.GELU()
            ))
        # 连接后的common 融合注意力机制
        trans_enc = nn.TransformerEncoderLayer(d_model=self.fusion_dim, nhead=1, dim_feedforward=1024,dropout=0.0)
        if self.use_up_and_down != 0:
            fusion_to_cluster = nn.Sequential(
                nn.Linear(self.fusion_dim, self.up_and_down_dim),
                # lusc
                nn.Linear(self.up_and_down_dim, 128),
                nn.ReLU(),
                nn.Linear(128, self.up_and_down_dim),
                nn.Linear(self.up_and_down_dim, self.cluster_var_dim)
            )
        else:
            fusion_to_cluster = nn.Sequential(
                nn.Linear(self.fusion_dim, self.fusion_dim),
                nn.Linear(self.fusion_dim, self.cluster_var_dim),
            )
        for i in range(view_num):
            decoder_MLP = []
            self.peculiar_and_class_dim = mu_logvar_dim + self.cluster_var_dim
            decoder_MLP += [
                nn.Linear(self.peculiar_and_class_dim, encoder_dim[-1]),
                nn.GELU()
            ]
            for j in range(len(encoder_dim) - 1):
                decoder_MLP += [
                    nn.Linear(encoder_dim[-(j + 1)], encoder_dim[-(j + 2)]),
                    nn.GELU()
                ]
            decoder_MLP += [
                nn.Linear(encoder_dim[0], in_dim[i]),
                nn.Sigmoid()
            ]
            Mv_decoder_MLP.append(nn.Sequential(*decoder_MLP))

        self.Mv_in_to_feature = nn.ModuleList(Mv_encoder_MLP)
        self.Mv_feature_peculiar = nn.ModuleList(Mv_feature_peculiar)
        self.Mv_peculiar_to_mu_logvar = nn.ModuleList(Mv_peculiar_to_mu_logvar)
        self.Mv_feature_to_common = nn.ModuleList(Mv_feature_to_common)
        self.Mv_common_to_fusion = nn.TransformerEncoder(trans_enc, num_layers=1)
        self.fusion_to_cluster = fusion_to_cluster
        self.Mv_decoder_MLP = nn.ModuleList(Mv_decoder_MLP)
        # 标签监督层
        # self.cluster_MLP = nn.Linear(self.cluster_num,self.cluster_num)
        self.cluster_layer = nn.Parameter(torch.zeros([self.cluster_num, self.cluster_var_dim], dtype = torch.float32))

    def encoder(self, X):
        '''
        :param X: 3 * b * d 三个视图 b是batch_size d是特征维度
        :return: mu,logvar,common
        '''
        mu = []
        logvar = []
        common = []
        Mv_common_peculiar = []
        for net_index in range(self.view_num):
            view_index = net_index
            feature = self.Mv_in_to_feature[net_index](X[view_index])
            peculiar = self.Mv_feature_peculiar[net_index](feature)
            mu.append(self.Mv_peculiar_to_mu_logvar[net_index](peculiar))
            logvar.append(self.Mv_peculiar_to_mu_logvar[net_index](peculiar))
            temp = self.Mv_feature_to_common[net_index](feature)  # 单个视图的common
            common.append(temp)
            Mv_common_peculiar.append([peculiar, temp])
        # # print(feature)
        Mv_common = torch.concat(common, dim=1)
        Mv_common = torch.unsqueeze(Mv_common, dim=1)
        fusion = self.Mv_common_to_fusion(Mv_common)
        fusion = fusion.reshape([Mv_common.shape[0], -1])
        cluster_var = self.fusion_to_cluster(fusion)

        return Mv_common_peculiar, fusion, cluster_var, mu, logvar

    def reparameterization(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn(std.shape).to(self.device)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def decoder(self, peculiar_and_common):
        out_list = []
        for i in range(self.view_num):
            temp = self.Mv_decoder_MLP[i](peculiar_and_common[i])
            out_list.append(temp)
        return out_list

    def forward(self, X):
        '''
        :param X: 3 * b * d
        :return:
        '''
        latent_dist = dict()
        Mv_common_peculiar, fusion, cluster_var, mu, logvar = self.encoder(X)
        # 连接Z 和 common
        peculiar_and_common = []
        z = []
        for i in range(self.view_num):
            # # 防止VAE坍塌
            bn = nn.BatchNorm1d(mu[i].shape[1]).to(self.device)
            mu[i] = bn(mu[i])
            logvar[i] = bn(logvar[i])
            z.append(self.reparameterization(mu[i], logvar[i]))
            peculiar_and_common.append(torch.concat([z[i], cluster_var], dim=1))

        out_list = self.decoder(peculiar_and_common)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(cluster_var.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        # print(self.cluster_layer)

        # 返回值列表
        latent_dist['mu_logvar'] = [mu, logvar]
        latent_dist['fusion'] = fusion
        latent_dist['Mv_common_peculiar'] = Mv_common_peculiar
        latent_dist['z'] = z  # mu,logvar 链接之后的z
        latent_dist['cluster_var'] = cluster_var
        latent_dist['q'] = q
        return out_list, latent_dist