# 继main1.py计算msd。。。

import tables as tb
import math
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import warnings
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D

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
path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_copy\\6.0\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！
filename = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)

path3 = 'D:\\guan2019\\2_ball\\3_ananlysis\\60Hz\\'
# translational
frame_traj = [path3 + 'trans_' + 'traj_' + name + '.jpg' for name in filename]  #
frame_msd = [path3 + 'trans_' + 'msd_' + name + '.h5' for name in filename]  #
frame_msd_pic = [path3 + 'trans_' + 'msd_' + name + '.jpg' for name in filename]  #
frame_pdf = [path3 + 'trans_' + 'pdf_' + name + '.h5' for name in filename]
frame_pdf_pic = [path3 + 'trans_' + 'pdf_' + name + '.jpg' for name in filename]  #
frame_posi = [path3 + 'trans_' + 'posi_' + name + '.jpg' for name in filename]  #
# rotational
rot_traj = [path3 + 'rot_' + 'traj_' + name + '.jpg' for name in filename]  #
rot_msd = [path3 + 'rot_' + 'msd_' + name + '.h5' for name in filename]  #
rot_msd_pic = [path3 + 'rot_' + 'msd_' + name + '.jpg' for name in filename]  #
rot_pdf = [path3 + 'rot_' + 'pdf_' + name + '.h5' for name in filename]
rot_pdf_pic = [path3 + 'rot_' + 'pdf_' + name + '.jpg' for name in filename]  #
rot_posi = [path3 + 'rot_' + 'posi_' + name + '.jpg' for name in filename]  #

# 位置和角度从_copy文件夹直接读
# frame_posi_trans = [path3 + 'posi_trans\\' + name + '.h5' for name in filename]
# frame_posi_rot = [path3 + 'posi_rot\\' + name + '.h5' for name in filename]

print(frame_traj, frame_msd)
print([i for i in range(len((1, 2, 3)))])

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

# translational-----------------------------------------------------------------------------------------

    # traj pic
    traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'x': center[:, 0], 'y': center[:, 1]})
    print(traj.head())
    ax = traj.plot(x='x', y='y', alpha=0.6, legend=False, title='trajectory')
    ax.set_xlim(0, 480)  # (traj['x'].min()-10, traj['x'].max()+10)
    ax.set_ylim(0, 480)  # (traj['y'].min()-10, traj['y'].max()+10)
    ax.set_xlabel('x(pixel)')
    ax.set_ylabel('y(pixel)')
    ax.plot()
    plt.show()
    fig = ax.get_figure()
    # fig.savefig(frame_traj[i])
    del ax, fig

    # msd
    dt = max_time/N
    msd = compute_msd(traj, t_step=dt, coords=['x', 'y'])
    print(msd.head())
    ax = msd.plot(x="tau", y="msds", logx=True, logy=True, legend=False, title='MSD')
    ax.fill_between(msd['tau'], msd['msds'] - msd['msds_std'], msd['msds'] + msd['msds_std'], alpha=0.2)
    ax.plot()
    plt.show()
    msd_i = pd.HDFStore(frame_msd[i], complib='blosc')
    msd_i.append(frame_name, msd, format='t', data_columns=True)
    fig = ax.get_figure()
    # fig.savefig(frame_msd_pic[i])
    del msd, msd_i, ax, fig

    # TODO:搁置了先，还没写完pdf
    deltax = center[1:, 0] - center[:-1, 0]  # numpy array
    deltay = center[1:, 1] - center[:-1, 1]
    r = np.sqrt(center[:, 0] ** 2 + center[:, 1] ** 2)
    deltar = r[1:] - r[:-1]
    # #
    # # weights_r = np.ones_like(deltar) / float(len(deltar))
    # # ax0.hist(deltar, 600, density=1, histtype='bar', facecolor='yellowgreen', weights=weights_r, alpha=0.75, rwidth=0.5)
    # # ax0.set_xlabel('deltaR(pixel)')
    # # ax0.set_ylabel('P')
    # # ax0.set_xlim(-2,2)#-max(-np.min(deltar), np.max(deltar)), max(-np.min(deltar), np.max(deltar)))
    # # ax0.plot()
    # # plt.show()
    # # # fig.savefig(frame_pdf_pic[i])
    # # del ax0, fig

    # trans
    x = center[:, 0]  # numpy array
    y = center[:, 1]
    time = np.linspace(0, max_time, N)
    plt.plot(time, r, color='yellowgreen')
    plt.title('translational position (pixel) vs time(s)')
    #TODO:没加坐标轴
    plt.savefig(frame_posi[i])
    plt.show()

# # rotational--------------------------------------------------------------------------------------------------------------
#
#     # ---- read the matrix and get the angles-----------------------------------------------------------------------------
#     points_matrix = np.reshape(np.array(matrix), (len(np.array(matrix)), 3, 3))
#     # print(matrix[0])  # 用来检查reshape输出的矩阵是否正确
#     # print(points_matrix)  # reshape的矩阵
#     # ---- 计算角度
#     # ---- # ---- 1个角一个轴
#     points_theta = np.array([np.arccos((np.ndarray.trace(x) - 1) / 2.) for x in
#                              points_matrix])  # points_theta = arccos((tr(matrix)-1)/2) 弧度
#     points_theta1 = np.array([[points_matrix[x][2, 1] - points_matrix[x][1, 2],
#                                points_matrix[x][0, 2] - points_matrix[x][2, 0],
#                                points_matrix[x][1, 0] - points_matrix[x][0, 1]] for x in range(len(points_matrix))])
#     points_axis = np.array([points_theta1[x] / (2 * math.sin(points_theta[x])) for x in range(
#         len(points_matrix))])  # axis = [R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]]/(2*SIN(THETA))
#     # print(points_axis)
#     # print(points_theta)
#     # ---- # ---- 欧拉角
#     points_euler = np.array([[math.atan2(x[2, 1], x[2, 2]),
#                               math.atan2(-x[2, 0], math.sqrt(x[2, 1] ** 2 + x[2, 2] ** 2)),
#                               math.atan2(x[1, 0], x[0, 0])] for x in points_matrix])  # euler 弧度 （旋转矩阵转欧拉角查公式
#     # print(points_euler)
#     # --------------------------------------------------------------------------------------------------------------------
#
#
#     # ---- calculate -----------------------------------------------------------------------------------------------------
#     # -- rot_traj pic
#     points_traj = pd.DataFrame(
#         {'t': np.linspace(0, max_time, N), 'x': points[:, 0], 'y': points[:, 1], 'z': points[:, 2]})
#     px = points_traj['x']
#     py = points_traj['y']
#     pz = points_traj['z']
#     # Make ball
#     br = np.floor(np.sqrt(px[0] ** 2 + py[0] ** 2 + pz[0] ** 2)) - 2
#     bu = np.linspace(0, 2 * np.pi, 1000)
#     bv = np.linspace(0, 2 * np.pi, 1000)
#     bx = br * np.outer(np.cos(bu), np.sin(bv))
#     by = br * np.outer(np.sin(bu), np.sin(bv))
#     bz = br * np.outer(np.ones(np.size(bu)), np.cos(bv))
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot_surface(bx, by, bz, color='y', alpha=0.1)
#     ax.plot(px, py, pz, color='black', label='traj')
#     plt.show()
#     fig = ax.get_figure()
#     # fig.savefig(rot_traj[i])
#     del ax, fig
#
#     # # # -- msd
#     # # dt = max_time/N
#     # # msd = compute_msd(traj, t_step=dt, coords=['x', 'y'])
#     # # print(msd.head())
#     # # ax = msd.plot(x="tau", y="msds", logx=True, logy=True, legend=False, title='MSD')
#     # # ax.fill_between(msd['tau'], msd['msds'] - msd['msds_std'], msd['msds'] + msd['msds_std'], alpha=0.2)
#     # # ax.plot()
#     # # plt.show()
#     # # msd_i = pd.HDFStore(frame_msd[i], complib='blosc')
#     # # msd_i.append(frame_name, msd, format='t', data_columns=True)
#     # # fig = ax.get_figure()
#     # # # fig.savefig(frame_msd_pic[i])
#     # # del msd, msd_i, ax, fig
#
#     # -- rot_pdf
#     # 点的r的pdf
#     pr = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2 + points[:, 2] ** 2)
#     deltapr = pr[1:] - pr[:-1]
#     ax0 = plt.subplot(111)
#     weights_r = np.ones_like(deltapr) / float(len(deltapr))
#     ax0.hist(deltapr, 6000, density=1, histtype='bar', facecolor='yellowgreen', weights=weights_r, alpha=0.75, rwidth=0.5)
#     ax0.set_xlabel('deltaR(pixel)')
#     ax0.set_ylabel('P')
#     # ax0.set_xlim(-0.1,0.1)#-max(-np.min(deltar), np.max(deltar)), max(-np.min(deltar), np.max(deltar)))
#     ax0.plot()
#     plt.show()
#     fig = ax0.get_figure()
#     # fig.savefig(rot_pdf_pic[i])
#     del ax0
#     # TODO:角度的pdf还没写
#
#     # # rot_trans
#     # # 只有角度的transition
#     # x = points[:, 0]  # numpy array
#     # y = points[:, 1]
#     # z = points[:, 2]
#     # time = np.linspace(0, max_time, N)
#     #
#     # weights_r = np.ones_like(deltapr) / float(len(deltapr))
#     # plt.plot(time, pr, '.', color='yellowgreen')
#     # # plt.set_xlabel('time(s)')
#     # # plt.set_ylabel('R(pixel)')
#     # plt.show()
#     # # fig.savefig(frame_posi[i])
#     # del fig
