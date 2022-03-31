# traj， fluctuation，log-fit 球心PDF！！！
#   'D:\\guan2019\\2_ball\\2_data\\60Hz_select\\' 是用来画translational pdf的！！！！！
#  'D:\\guan2019\\2_ball\\2_data\\60Hz_select_aucorre\\'  # 画自相关

import tables as tb
import math
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import warnings
import matplotlib.cm as cm
from scipy.spatial.distance import cdist
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets    #导入内置数据集模块
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller as ADF  #平稳性检测
from statsmodels.stats.diagnostic import acorr_ljungbox    #白噪声检验
from statsmodels.tsa.arima_model import ARIMA
from scipy.optimize import curve_fit
import math

warnings.filterwarnings('ignore')
# TODO: CHANGE PARAMETERS HERE------------------
fps = 150.0


# -----------------------------------------------
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def func(x, a, b, c):
    return (a*np.exp(-b*np.abs(x)**c))

def funce(x, a, b, c):
    return (a*np.exp(-b*x**c))


def smooth(x, window_len=51, window='flat'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y

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

def Pettitt_change_point_detection(inputdata):
    inputdata = np.array(inputdata)
    n         = inputdata.shape[0]
    k = range(n)
    inputdataT = pd.Series(inputdata)
    r = inputdataT.rank()
    Uk = [2*np.sum(r[0:x])-x*(n + 1) for x in k]
    Uka = list(np.abs(Uk))
    U = np.max(Uka)
    K = Uka.index(U)
    pvalue         = 2 * np.exp((-6 * (U**2))/(n**3 + n**2))
    if pvalue <= 0.05:
        change_point_desc = '显著'
    else:
        change_point_desc = '不显著'
    #Pettitt_result = {'突变点位置':K,'突变程度':change_point_desc}
    return K #,Pettitt_result


# read position from hdf5 file
path2 = 'D:\\guan2019\\2_ball\\2_data\\60tt\\'#60Hz_select\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！#
# path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_aucorre\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！#

filename = [name for name in os.listdir(path2)]
pdf_trans_dict = {}
pdf_transenergy_dict = {}
center_all = {}
delta_all = {}
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for j in range(len(filename)):  #3,4):    #
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3)]
    file_n = [path3 + name + '.h5' for name in filename1]
    print(filename1, file_n)

    d_all = []  # 合在一起的dx和dy

    for i in range(len(file_n)):  # 2,3):    #   # TODO 在len(file_n)):  #这里更改文件，把同意条件下dx dy合在一起计算pdf
        store = pd.HDFStore(file_n[i], mode='r')
        print(store.keys())
        center = store.get('center').values[::1]  # numpy array
        store.close()

        center = center# [::30]  # 大的timestep

        frame_name = filename1[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
        # print(type(frame_name))

        delta = np.diff(center, axis=0)
        # print(center.shape)
        # print(np.min(delta),np.max(delta))
        index = np.where(abs(delta)>25)
        # print(index[0])
        # print(delta[index])

# 用1/150s的时候用这个筛选
        if index[0] != ():
            print('1')
        print(len(index[0]))
        for k in range(len(index[0])-1):
            if k >= len(index[0]):
                break
            center = np.delete(center, index[0][0], 0)
            center = np.delete(center, index[0][0], 0)
            center = np.delete(center, index[0][0], 0)
            # print(center.shape)
            delta = np.diff(center, axis=0)
            index = np.where(abs(delta) > 20)
            # print(index[0])
            # print(delta[index])


        step = 1 #!!!!!!!!!!
        delta_a = np.array([np.sum(delta[i:i + step,0]) for i in range(0, len(delta[:,0]), step)])
        delta_b = np.array([np.sum(delta[i:i + step, 1]) for i in range(0, len(delta[:, 1]), step)])
        delta = np.vstack((delta_a, delta_b)).T  # 更大步长的delta
        center_a = np.array([np.average(center[i:i + step,0]) for i in range(0, len(center[:,0]), step)])
        center_b = np.array([np.average(center[i:i + step, 1]) for i in range(0, len(center[:, 1]), step)])
        center = np.vstack((center_a, center_b)).T  # 更大步长的center





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



        # 找数据断点 （待改）
        # dt = delta[:,0]
        # print("Pettitt:", Pettitt_change_point_detection(dt))
        # print("Mann-Kendall:", Kendall_change_point_detection(dt))
        #
        # print("Mann-Kendall:", Kendall_change_point_detection(dt))
        # print("Pettitt:", Pettitt_change_point_detection(dt))
        # print("Buishand U Test:", Buishand_U_change_point_detection(dt))
        # print("Standard Normal Homogeneity Test (SNHT):", SNHT_change_point_detection(dt))


        N = len(center[:,0])
        max_time = N / fps * step # seconds
        time = np.linspace(0, N-1, N) *1/150.0 # (0, max_time, N)
        dt = max_time/N

        d_all = np.hstack((d_all,delta[:,0]))
        # print(d_all.shape)
        d_all = np.hstack((d_all,delta[:,1]))
        print(d_all.shape)






    # 
    # # -------每个文件traj pic--------------------
    #     traj = pd.DataFrame({'t': time, 'x': center[:,0], 'y': center[:,1]})
    #     # print(traj.head())
    #     ax = traj.plot(x='x', y='y', alpha=0.6,lw=0.5, legend=False, title='trajectory')
    #     ax.set_xlim(0, 480)  # (traj['x'].min()-10, traj['x'].max()+10)
    #     ax.set_ylim(0, 480)  # (traj['y'].min()-10, traj['y'].max()+10)
    #     ax.set_xlabel('x(pixel)')
    #     ax.set_ylabel('y(pixel)')
    #     ax.plot()
    #     plt.text(1.5, 1, '1 pixel=0.54 mm', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
    #     plt.show()
    #     fig = ax.get_figure()
    #     del ax, fig



    # # -------x y -- fluctuation--------------------
    #     fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6))
    #     ax0 = plt.subplot(211)
    #     # plt.plot(time, center[:, 0], lw=0.5, marker='o', markersize=0.7)  # Displacement Position pixel  # mm / 480.0 * 260
    #     plt.scatter(time, center[:, 0], s=3) # Displacement Position pixel # mm / 480.0 * 260
    #
    #     ax0.set_xlabel('time (s)')
    #     ax0.set_ylabel('$x (pixel)$')
    #     ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    #     plt.title('Position' + '[60Hz, ' + str(frame_name) + '$g$]', fontsize=10)  #  Displacement  Position
    #
    #     ax1 = plt.subplot(212)
    #     plt.scatter(time, center[:, 1], s=3)  # pixel
    #     ax1.set_xlabel('time (s)')
    #     ax1.set_ylabel('$y (pixel)$')
    #     ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    #     plt.show()


    # # -------delta x delta y -- fluctuation--------------------
    #     fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6))
    #     ax0 = plt.subplot(211)
    #     # plt.plot(time, center[:, 0], lw=0.5, marker='o', markersize=0.7)  # Displacement Position pixel  # mm / 480.0 * 260
    #     plt.scatter(time[1:], np.diff(center[:, 0]), s=3) # Displacement Position pixel # mm / 480.0 * 260
    #
    #     ax0.set_xlabel('time (s)')
    #     ax0.set_ylabel('$\Delta x (pixel)$')
    #     ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    #     plt.title('Fluctuation' + '[60Hz, ' + str(frame_name) + '$g$]', fontsize=10)  #  Displacement  Position
    #     ax0.set_xlim(30,70)
    #     ax0.set_ylim(-1,1)
    #
    #     ax1 = plt.subplot(212)
    #     plt.scatter(time[1:], np.diff(center[:, 1]), s=3)  # pixel
    #     ax1.set_xlabel('time (s)')
    #     ax1.set_ylabel('$\Delta y (pixel)$')
    #     ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    #     ax1.set_xlim(30, 70)
    #     ax1.set_ylim(-1, 1)
    #     plt.show()


    center_all[filename[j] + 'x'] = np.sqrt(delta[:, 0]**2+delta[:,1]**2)  # center[:, 0]   #   np.sqrt(center[:, 0]**2+center[:,1]**2)  # delta[:,0]  #
    center_all[filename[j] + 'y'] = center[:, 1]
    delta_all[filename[j] + 'x'] = delta[:, 0]
    delta_all[filename[j] + 'y'] = delta[:, 1]


# -------------------d_all互相关---------
#         v_t1 = delta[:,0]
#         v_t2 = delta[:, 1]
#         v_t1 = (v_t1 - np.mean(v_t1)) / (np.std(v_t1) * len(v_t1))
#         v_t2 = (v_t2 - np.mean(v_t2)) / (np.std(v_t2))
#         corr = np.correlate(v_t1, v_t2, 'full')
#
#         plt.figure()
#         # plt.plot(time[::50],smooth(corr)[:len(time):50],'.-', markerfacecolor='none',alpha=0.3)
#         plt.plot(time, corr[:len(time)], 'o',alpha=0.65, color=colors[i], label=frame_name + 'Hz')
#         plt.xscale('log')
#         plt.show()



#------计算所有文件PDF--------------------

    AU, BU, CU = plt.hist(d_all/ 480.0 * 260, np.arange(-1.025,1.05,0.05)/ 480.0 * 260, histtype='bar', facecolor='blue', alpha=0.75, rwidth=0.2)  # , density=True)
    AU /= len(d_all)
    AU_ind = np.where(AU == 0)
    AU = np.delete(AU, AU_ind)
    AU1 = np.array([math.log(x) for x in AU])
    pdf_trans_dict[filename[j] + 'y'] = AU1
    BU = (BU[:-1] + BU[1:]) / 2.
    BU = np.delete(BU, AU_ind)
    pdf_trans_dict[filename[j] + 'x'] = BU  # 存入dict

# energy---------
    aue, bue, cue = plt.hist(0.0097/2.0 * (center_all[filename[j] + 'x']*150/ 480.0 * 0.26) ** 2 , np.arange(0,0.00002,0.000001), histtype='bar',facecolor='yellowgreen', alpha=0.75, rwidth=1)  # , density=True)  # au是counts，bu是deltar
    aue /=  len(center_all[filename[j] + 'x'])
    aue_ind = np.where(aue == 0)
    aue = np.delete(aue, aue_ind)
    aue1 = np.array([math.log(x) for x in aue])
    pdf_transenergy_dict[filename[j] + 'y'] = aue1
    bue = (bue[:-1] + bue[1:]) / 2.
    bue = np.delete(bue, aue_ind)
    pdf_transenergy_dict[filename[j] + 'x'] = bue  # 存入dict

# # trans - pdf----------------------------------------------------------------------------------------------------------------
ys = [i +  (i ) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

fig = plt.figure()
ax1 = fig.add_subplot(111)
label = []
for i in range(len(filename)):  #  2,3):    #
    # plt.plot(pdf_trans_dict[filename[i]+'x'] , pdf_trans_dict[filename[i]+'y'],'o-',alpha=0.5,color=colors[i])#, cmap='hsv')
    ax1.scatter(pdf_trans_dict[filename[i] + 'x'], np.exp(pdf_trans_dict[filename[i] + 'y']), alpha=0.75,
                color=colors[i])  # , label=label[i])

    label.append(filename[i] + 'g')
ax1.legend(label)

# #
# # ########## Translatioanl fitting
# for i in range(len(filename)):  #  2,3):    #
#     middle = np.where(pdf_trans_dict[filename[i]+'x']  == find_nearest(pdf_trans_dict[filename[i]+'x'], 0))[0][0]  # pdf 0 的点
#     x2 = pdf_trans_dict[filename[i]+'x'][:middle+1]  # 越过中间的几个点
#     y2 = pdf_trans_dict[filename[i]+'y'][:middle+1]
#     x1 = -np.flipud(pdf_trans_dict[filename[i]+'x'][middle:])
#     y1= np.flipud(pdf_trans_dict[filename[i]+'y'][middle:])
#
#     print(len(x1),len(x2))
#     n = np.min((len(x2),len(x1)))
#     x2 = x2[len(x2)-n:]
#     x1 = x1[len(x1)-n:]
#     y2 = y2[len(y2)-n:]
#     y1 = y1[len(y1)-n:]
#     x = np.array((x1+x2)/2.0)[5:]
#     y = np.array((y1+y2)/2.0)[5:]-np.min(pdf_trans_dict[filename[i] + 'y'])
#     popt, pcov = curve_fit(func, x, y, maxfev=1000)
#     y11 = [func(p, popt[0],popt[1],popt[2]) for p in x]
#     print(filename[i])
#     print(popt)
#     print('---')
#     plt.plot(-np.flipud(x),np.flipud(np.exp(y11+np.min(pdf_trans_dict[filename[i] + 'y']))),color=colors[i], alpha=0.5)
#
#
# leg = plt.legend(label)
# leg.get_frame().set_linewidth(0.0)
# ax1.set_title('Translational PDF' + ' [60Hz] ', fontsize=10)
# ax1.set_xlabel('$\Delta x(pixel)$')
# ax1.set_ylabel('$P_{\Delta x}$')
# plt.yscale('log')
# ax1.set_ylim(0.0001, 100)
# plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)
#
# plt.show()

# trans - energy ----------------------------------------------
fig = plt.figure()
ax = fig.add_subplot(111)
for i in range(len(filename)):#2,3):#
    ax.scatter(pdf_transenergy_dict[filename[i] + 'x']*1000, np.exp(pdf_transenergy_dict[filename[i] + 'y']),alpha=0.65,color=colors[i],label=frame_name + 'g')
    # ax.plot(pdf_transenergy_dict[filename[i] + 'x']*1000000, np.exp(pdf_transenergy_dict[filename[i] + 'y']), alpha=0.65,lw=1, color=colors[i],label=frame_name + 'g')

# ########## Translatioanl fitting
for i in range(len(filename)):  #  2,3):    #
    if filename[i] == '4.5':
        x = pdf_transenergy_dict[filename[i] + 'x'][:-6] * 1000
        y = pdf_transenergy_dict[filename[i] + 'y'][:-6] - np.min(pdf_transenergy_dict[filename[i] + 'y'])
    elif filename[i] == '6.0':
        x = pdf_transenergy_dict[filename[i] + 'x'][:-5] * 1000
        y = pdf_transenergy_dict[filename[i] + 'y'][:-5] - np.min(pdf_transenergy_dict[filename[i] + 'y'])
    elif filename[i] == '5.5':
        x = pdf_transenergy_dict[filename[i] + 'x'][:-2] * 1000
        y = pdf_transenergy_dict[filename[i] + 'y'][:-2] - np.min(pdf_transenergy_dict[filename[i] + 'y'])
    else:
        x = pdf_transenergy_dict[filename[i] + 'x'][:-3]*1000 # -3 4.5/5.5/6/ no
        y = pdf_transenergy_dict[filename[i] + 'y'][:-3]-np.min(pdf_transenergy_dict[filename[i] + 'y'])
    popt, pcov = curve_fit(func, x, y,maxfev=10000000)
    x1 = pdf_transenergy_dict[filename[i] + 'x']*1000
    y11 = [func(p, popt[0],popt[1],popt[2]) for p in x1]
    print(filename[i])
    print(popt)
    print('---')
    ax.plot(x1,np.exp(y11+np.min(pdf_transenergy_dict[filename[i] + 'y'])),color=colors[i], alpha=0.5)

# ax.set_ylim((0.0006, 1))
ax.set_xlabel(r'${E_T~(mJ)}$')
ax.set_ylabel(r'${P(E_T)}$')
ax.set_yscale('log')
ax.legend(label)
leg = ax.legend(label)
leg.get_frame().set_linewidth(0.0)


plt.show()


