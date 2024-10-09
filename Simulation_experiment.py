import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.stats._mstats_basic import winsorize

seed = 5
np.random.seed(seed)
# M * N
def construct_simulation_data(times):
    exp = pd.read_csv("./data/real_data/GSE_exp.csv",sep=',').dropna()
    methy = pd.read_csv("./data/real_data/GSE_methy.csv",sep=',').dropna()
    mirna = pd.read_csv("./data/real_data/GSE_mirna.csv",sep=',').dropna()

    sub_exp = np.mat(exp.sample(n=400,axis=1).to_numpy())
    sub_methy = np.mat(methy.sample(n=400,axis=1).to_numpy())
    sub_mirna = np.mat(mirna.sample(n=400,axis=1).to_numpy())

    U1, S1, VT1 = la.svd(sub_exp,full_matrices=False)
    U2, S2, VT2 = la.svd(sub_methy, full_matrices=False)
    U3, S3, VT3 = la.svd(sub_mirna, full_matrices=False)
    # print(np.shape(U_exp),np.shape(np.diag(sigma_exp)),np.shape(VT_exp))

    # 生成随机加入的mean and value
    exp_sim = np.random.normal(0,3,size=np.shape(sub_exp))
    methy_sim = np.random.normal(0,3,size=np.shape(sub_methy))
    mirna_sim = np.random.normal(0,3,size=np.shape(sub_mirna))

    mean = np.random.permutation([0, 0.25, 0.5, 0.75])
    exp_sim[:,0:100] += mean[0]
    exp_sim[:,100:300] += mean[1]
    exp_sim[:,300:400] += mean[2]

    mean = np.random.permutation([0, 0.25, 0.5, 0.75])
    methy_sim[:,0:100] += mean[0]
    methy_sim[:,200:300] += mean[0]
    methy_sim[:,100:200] += mean[1]
    methy_sim[:,300:400] += mean[2]

    mean = np.random.permutation([0, 0.25, 0.5, 0.75])
    mirna_sim[:,0:200] += mean[0]
    mirna_sim[:,200:300] += mean[1]
    mirna_sim[:,300:400] += mean[2]

    _, _, VT_sim1 = la.svd(exp_sim, full_matrices=False)
    _, _, VT_sim2 = la.svd(methy_sim, full_matrices=False)
    _, _, VT_sim3 = la.svd(mirna_sim, full_matrices=False)

    exp_sim = U1 @ np.diag(S1) @ VT_sim1
    methy_sim = U2 @ np.diag(S2) @ VT_sim2
    mirna_sim = U3 @ np.diag(S3) @ VT_sim3

    simData = [exp_sim,methy_sim,mirna_sim]
    name_list = ["exp","methy","mirna"]

    noiseqd = 0.3
    for i in range(3):
        noise_amplitude = noiseqd * simData[i]
        noise = np.multiply(np.random.normal(size=simData[i].shape),noise_amplitude)
        simData[i] += noise
        np.savetxt('./data/simdata3/{}_{}.csv'.format(name_list[i],times), simData[i], fmt='%f', delimiter=',')

for i in range(50):
    construct_simulation_data(i)


# fig = plt.figure(figsize=[12,4])
# for i in range(3):
#     ax = fig.add_subplot(1,3,i+1)
#     ax.imshow(simData[i][:400,:])
#     ax.set_xlabel("Sample",fontsize=14)
#     ax.set_ylabel("Feature",fontsize=14)
#     ax.set_title(name_list[i],fontsize=14)
#     ax.set_xticks([])
#     ax.set_yticks([])
#
# plt.show()
# plt.tight_layout() #去除pdf周围白边
# plt.savefig("./figs/sim_data.pdf")
# print(simData_exp.shape)
# np.savetxt('./data/simdata/exp.csv',simData_exp,fmt='%f',delimiter=',') #frame: 文件 array:存入文件的数组
# np.savetxt('./data/simdata/methy.csv',simData_methy,fmt='%f',delimiter=',') #frame: 文件 array:存入文件的数组
# np.savetxt('./data/simdata/mirna.csv',simData_mirna,fmt='%f',delimiter=',') #frame: 文件 array:存入文件的数组