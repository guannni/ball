#  与main1_rot_v.py一样，不过用于画fluctuation，autocorrelation，处理四元数,以及 axis-angle中theta，
#   'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\' 是用来画rotational pdf的！！！！！

import tables as tb
import math
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import warnings
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

warnings.filterwarnings('ignore')
# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0


# -----------------------------------------------

def compute_msd(trajectory, t_step, coords=['x', 'y']):
    tau = trajectory['t'].copy()
    shifts = np.floor(tau / t_step).astype(np.int)
    msds = np.zeros(shifts.size)
    msds_std = np.zeros(shifts.size)

    for i, shift in enumerate(shifts):
        diffs = trajectory[coords] - trajectory[coords].shift(-shift)
        sqdist = np.square(diffs).sum(axis=1)
        msds[i] = sqdist.mean()
        msds_std[i] = sqdist.std()

    msds = pd.DataFrame({'msds': msds, 'tau': tau, 'msds_std': msds_std})
    return msds


def traversalDir_FirstDir(path):  # 返回一级子文件夹名字
    list = []
    if (os.path.exists(path)):  # 获取该目录下的所有文件或文件夹目录路径
        files = glob.glob(path + '\\*')
        # print(files)
        for file in files:  # 判断该路径下是否是文件夹
            if (os.path.isdir(file)):  # 分成路径和文件的二元元组
                h = os.path.split(file)
                print(h[1])
                list.append(h[1])
        return list



# read position from hdf5 file
path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！

filename = [name for name in os.listdir(path2)]
pdf_rot_dict = {}

for j in range(len(filename)):  #3,4):    # len(filename)-1,
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3)]
    file_n = [path3 + name + '.h5' for name in filename1]
    print(filename1, file_n)

    d_theta = []
    d_pr = []
    d_axis = [[0,0,0]]
    d_euler = [[0,0,0]]
    d_quaternions = [[0,0,0,0]]


    for i in range(len(file_n)):
        store = pd.HDFStore(file_n[i], mode='r')
        print(store.keys())
        center = store.get('center').values  # numpy array
        matrix = store.get('matrix').values
        points = store.get('points').values
        store.close()

        N = len(center)
        max_time = N / fps  # seconds
        frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
        print(type(frame_name))


    # rotational--------------------------------------------------------------------------------------------------------------
        points_reshape = np.reshape(points, (len(points), 6, 3))  # points 2维，points_reshape 3维
        matrix_reshape = np.reshape(np.array(matrix), (len(np.array(matrix)), 3, 3))  # reshape的矩阵
        points_1 = points_reshape[0][0]
        print(points_1)
        for k in range(1,len(points)):
            # print(np.linalg.norm(points_reshape[k][0]-points_1))
            # print(points_1)
            if np.linalg.norm(points_reshape[k][0] - points_1) > 10:
                # print(points_reshape[k][0],k)
                points_reshape[k] = points_reshape[k - 1]
                matrix_reshape[k - 1] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                matrix_reshape[k] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

            points_1 = points_reshape[k][0]


        # # ---- 计算角度
        # # -------- 瞬时角+瞬时轴+欧拉角+四元数 --计算
        points_theta = np.nan_to_num(np.array([np.arccos((np.ndarray.trace(x) - 1) / 2.) for x in
                                 matrix_reshape]) ) # points_theta = arccos((tr(matrix)-1)/2) 弧度
        points_theta1 = np.array([[matrix_reshape[x][2, 1] - matrix_reshape[x][1, 2],
                                   matrix_reshape[x][0, 2] - matrix_reshape[x][2, 0],
                                   matrix_reshape[x][1, 0] - matrix_reshape[x][0, 1]] for x in range(len(matrix_reshape))])  # 瞬时角 弧度
        points_axis = np.array([points_theta1[x] / (2 * math.sin(points_theta[x])) for x in range(
            len(matrix_reshape))])  # 瞬时轴 axis = [R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]]/(2*SIN(THETA))
        deltaeuler = np.array([[math.atan2(x[2, 1], x[2, 2]),
                                  math.atan2(-x[2, 0], math.sqrt(x[2, 1] ** 2 + x[2, 2] ** 2)),
                                  math.atan2(x[1, 0], x[0, 0])] for x in matrix_reshape])  # euler 弧度 （旋转矩阵转欧拉角查公式

        quaternions = np.array([[math.sqrt(np.ndarray.trace(matrix_reshape[x])+1)/2.0,
                                 (matrix_reshape[x][1, 2] - matrix_reshape[x][2, 1]) / 2.0 / math.sqrt(
                                     np.ndarray.trace(matrix_reshape[x]) + 1),
                                 (matrix_reshape[x][2, 0] - matrix_reshape[x][0, 2]) / 2.0 / math.sqrt(
                                     np.ndarray.trace(matrix_reshape[x]) + 1),
                                 (matrix_reshape[x][0, 1] - matrix_reshape[x][1, 0]) / 2.0 / math.sqrt(
                                     np.ndarray.trace(matrix_reshape[x]) + 1)] for x in range(len(matrix_reshape))])

        # ------------瞬时角+瞬时轴+欧拉角 --处理
        deltatheta = points_theta / math.pi * 180.0
        deltaeuler = deltaeuler / math.pi * 180.0  # euler角度

        all = np.vstack((deltatheta, points_axis[:, 0], points_axis[:, 1],
                         points_axis[:, 2], deltaeuler[:, 0], deltaeuler[:, 1], deltaeuler[:, 2], quaternions[:, 0],
                         quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]))  # theta[0,:],axis[1:4,:],euler[4:7,:],quaterions[7:11,:]


        index = np.where(deltatheta == 0)
        all = np.delete(all, index[0],axis=1)
        deltaaxis = np.delete(points_axis, index[0],axis=0)  # 瞬时轴
        index = np.where(np.nan_to_num(deltaaxis[:,0] == 0))
        all = np.delete(all, index[0],axis=1)

        index = []
        for k in range(1,len(all[0,:])):
            if (all[:,k]==all[:,k-1]).all():
                index.append(k)

        all = np.delete(all, index,axis=1)

        deltatheta = all[0,:].T  # 每次旋转的角度 (degree)
        deltaaxis = all[1:4,:].T  # 瞬时轴
        deltaeuler = all[4:7,:].T # euler (degree)
        quaternions = all[7:11,:].T  # quaternions

        d_theta = np.hstack((d_theta, deltatheta))
        d_axis = np.vstack((d_axis, deltaaxis))[1:,:]  # 去掉一开始那个[0,0,0]
        d_euler = np.vstack((d_euler, deltaeuler))[1:,:]  # 去掉一开始那个[0,0,0]
        d_quaternions = np.vstack((d_quaternions, quaternions))[1:,:]  # 去掉一开始那个[0,0,0]

# # quanternions-----------------------------------------------------------------------------------
        # quaternions-time(S)
        plt.figure()
        time = np.around(np.arange(0, len(quaternions) * 1 / 150, 1 / 150), decimals=2)
        ax1 = plt.subplot(221)
        plt.scatter(time,quaternions[:, 0], alpha=0.5, marker='.')
        ax2 = plt.subplot(222)
        plt.scatter(time,quaternions[:, 1], alpha=0.5, marker='.')
        ax3 = plt.subplot(223)
        plt.scatter(time,quaternions[:, 2], alpha=0.5, marker='.')
        ax4 = plt.subplot(224)
        plt.scatter(time,quaternions[:, 3], alpha=0.5, marker='.')
        plt.show()

       # quaternions自相关
        plt.figure()
        time = np.around(np.arange(0, len(quaternions) * 1 / 150, 1 / 150), decimals=2)
        ax1 = plt.subplot(221)
        plot_acf(quaternions[:,0], alpha=0.05, lags=time, marker='+', markersize=0.5, use_vlines=False, zero=True,ax=ax1,title=' ')
        ax2 = plt.subplot(222)
        plot_acf(quaternions[:, 1], alpha=0.05, lags=time, marker='+', markersize=0.5, use_vlines=False, zero=True,ax=ax2,title=' ')
        ax3 = plt.subplot(223)
        plot_acf(quaternions[:, 2], alpha=0.05, lags=time, marker='+', markersize=0.5, use_vlines=False, zero=True,ax=ax3,title=' ')
        ax4 = plt.subplot(224)
        plot_acf(quaternions[:, 3], alpha=0.05, lags=time, marker='+', markersize=0.5, use_vlines=False, zero=True,ax=ax4,title=' ')
        plt.show()

# # theta------------------------------------------------------------------
#         # theta- time(S)
#         plt.figure()
#         time = np.around(np.arange(0, len(deltatheta) * 1 / 150, 1 / 150), decimals=2)
#         ax1 = plt.subplot(111)
#         plt.scatter(time,deltatheta, alpha=0.5, marker='.')
#         ax1.set_ylabel('$\Delta \Theta(°)$')
#         ax1.set_xlabel('time(s)')
#         plt.show()
#
#        # theta自相关
#         plt.figure()
#         time = np.around(np.arange(0, len(deltatheta) * 1 / 150, 1 / 150), decimals=2)
#         ax1 = plt.subplot(111)
#         plot_acf(deltatheta, alpha=0.05, lags=time, marker='+', markersize=0.5, use_vlines=False, zero=True,title=' ')
#         ax1.set_xlabel('time(s)')
#         plt.show()


