# 读取hdf文件，计算MSD
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import matplotlib.mlab as mlab
import seaborn as sns

# TODO: CHANGE PARAMETERS HERE------------------
fps = 240.0


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

    print(msds[0])
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
path2 = 'G:\\ball_new\\2_data\\frame\\20hz_240_copy\\'  # 'D:\\guan2019\\2_ball\\2_data_new\\60Hz\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！
filename = [name for name in os.listdir(path2) ]

# path4 = 'D:\\guan2019\\2_ball\\3_ananlysis\\trans\\msd\\60Hz_tt\\'
path4 = 'G:\\ball_new\\3_analysis\\msd\\trans\\'   # 'D:\\guan2019\\2_ball\\3_ananlysis_new\\trans_msd\\60Hz\\'

# pdf_trans_dict = {}
for j in range(2,3):    #
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3)]
    file_n = [path3 + name + '.h5' for name in filename1]
    frame_msd = [path4 + name + '.h5' for name in filename1]
    frame_msd_pic = [path4 + name + '.png' for name in filename1]
    print(filename1, file_n)


    for i in range(len(file_n)):#(0,2): # 3,4):#
        store = pd.HDFStore(file_n[i], mode='r')
        print(file_n[i])
        center = store.get('center').values  # numpy array
        store.close()

        frame_name = filename1[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
        # print(type(frame_name))
        delta = np.diff(center, axis=0)


        print(center.shape)
        print(np.min(delta), np.max(delta))
        index = np.where(abs(delta) > 100)
        print(index[0])
        print(delta[index])

        # if len(index[0]) != 0:
        #     print('1')
        # print(len(index[0]))
        # for k in range(len(index[0])):
        #     if k >= len(index[0]):
        #         break
        #     center = np.delete(center, index[0][0], 0)
        #     center = np.delete(center, index[0][0], 0)
        #     center = np.delete(center, index[0][0], 0)
        #     # print(center.shape)
        #     delta = np.diff(center, axis=0)
        #     index = np.where(abs(delta) > 20)
        #     # print(index[0])
        #     # print(delta[index])


        # while(np.max(abs(delta[:,0])) > 11):
        #     index1 = np.array(np.where(abs(delta[:,0]) > 11)[0])[::2] + 1
        #     # print(index1)
        #     center = np.delete(center, index1,0)
        #     # print(center.shape)
        #     delta = np.diff(center, axis=0)
        #     # print(np.max(delta[:,1]))
        # while(np.max(abs(delta[:,1])) > 11):
        #     index1 = np.array(np.where(abs(delta[:,1]) > 11)[0])[::2] + 1
        #     # print(index1)
        #     center = np.delete(center, index1,0)
        #     # print(center.shape)
        #     delta = np.diff(center, axis=0)
        #     # print(np.max(delta[:,1]))

        N = len(center[:,0])
        max_time = N / fps  # seconds
        time = np.linspace(0, max_time, N)
        dt = max_time/N

        # traj = pd.DataFrame({'t': time, 'x': center[:, 0], 'y': center[:, 1]})
        traj = pd.DataFrame({'t': time[:len(delta[:,0])], 'x': delta[:, 0], 'y': delta[:, 1]})
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
        # fig = ax.get_figure()
        # fig.savefig(frame_msd_pic[i])
        # del msd, msd_i, ax, fig
        del msd, msd_i

