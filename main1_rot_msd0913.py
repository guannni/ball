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
path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_copy\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！

# path4 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(axisangle)_msd\\' 
# path4 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(euler3)_msd\\' 
path4 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(qua4)_msd\\' 

filename = [name for name in os.listdir(path2)]

pdf_rot_dict = {}

for j in range(len(filename)):  #3,4):    # 
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3)]
    file_n = [path3 + name + '.h5' for name in filename1]
    frame_msd = [path4 + name + '.h5' for name in filename1]
    frame_msd_pic = [path4 + name + '.png' for name in filename1]
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

# 求theta的MSD
        N = len(deltatheta)
        max_time = N / fps  # seconds
        time = np.linspace(0, max_time, N)
        dt = max_time/N

        # traj = pd.DataFrame({'t': time, 'x': np.cumsum(deltatheta), 'y': np.zeros(len(deltatheta))})  # angle-axis method -- theta
        # traj = pd.DataFrame({'t': time, 'x': np.cumsum(deltaeuler[:, 2]), 'y': np.zeros(len(deltaeuler[:, 0]))})  # euler method -- theta
        traj = pd.DataFrame({'t': time, 'x': quaternions[:, 3], 'y': np.zeros(len(deltaeuler[:, 0]))})  # euler method -- theta

        print(type(frame_name))

        # msd
        msd = compute_msd(traj, t_step=dt, coords=['x', 'y'])
        print(msd.head())
        ax = msd.plot(x="tau", y="msds", logx=True, logy=True, legend=False, title='MSD')
        ax.fill_between(msd['tau'], msd['msds'] - msd['msds_std'], msd['msds'] + msd['msds_std'], alpha=0.2)
        ax.plot()
        # plt.show()
        msd_i = pd.HDFStore(frame_msd[i], complib='blosc')
        msd_i.append(frame_name, msd, format='t', data_columns=True)
        msd_i.close()
        # fig = ax.get_figure()
        # fig.savefig(frame_msd_pic[i])
        # del msd, msd_i, ax, fig
        del msd, msd_i


