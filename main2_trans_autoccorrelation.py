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
import scipy.fftpack
from scipy.signal import hilbert, chirp

warnings.filterwarnings('ignore')
# TODO: CHANGE PARAMETERS HERE------------------
fps = 240.0


# -----------------------------------------------
def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx]

def func(x, a, b, c):
    return (a*np.exp(-b*np.abs(x)**c))

def funce(x, a, b, c):
    return (a*np.exp(-b*x**c))

def autocorr(array):  # array is deltax
    # array *= 10000
    ac = np.zeros(len(array))
    ave = sum(array)/float(len(array))
    for i in range(len(array)):  # lags
        temp = []
        for j in range(i,len(array)):  # range(len(array)):
            if j + i < len(array):
                temp.append((array[j]-ave)*(array[j+i]-ave))  # (array[j]*array[j+i])
        # print(temp)
        temp /= np.var(array)  # np.square(array[:len(temp)])
        nan = np.isnan(np.array(temp))
        temp = np.delete(temp,nan)
        ac[i] = temp.sum()/len(temp)
    return ac   


def smooth(x, window_len=10, window='flat'):
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
# path2 = 'D:\\guan2019\\2_ball\\2_data\\60tt\\'#60Hz_select\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！#
path2 = 'G:\\ball_new\\3_analysis\\autocorr\\' 
# 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_acf\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！#


filename1 = [os.path.splitext(name)[0] for name in os.listdir(path2)]
file_n = [path2 + name + '.h5' for name in filename1]

print(filename1, file_n)


fig = plt.figure()
ax = fig.add_subplot(111)
label = []
pdf_trans_dict = {}
center_all = {}
delta_all = {}
ys = [i + (i) ** 2 for i in range(len(file_n))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for j in range(len(file_n)):  #3,4):    #

    d_all = []  # 合在一起的dx和dy

    store = pd.HDFStore(file_n[j], mode='r')
    print(store.keys())
    center = store.get('center').values  # numpy array
    store.close()

    center = center# [::30]  # 大的timestep

    frame_name = filename1[j].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    # print(type(frame_name))

    delta = np.diff(center, axis=0)
    # print(center.shape)
    # print(np.min(delta),np.max(delta))
    index = np.where(abs(delta)>25)
    # print(index[0])
    # print(delta[index])

# # 用1/150s的时候用这个筛选
#     if index[0] != ():
#         print('1')
#     print(len(index[0]))
#     for k in range(len(index[0])):
#         if k >= len(index[0]):
#             break
#         center = np.delete(center, index[0][0], 0)
#         center = np.delete(center, index[0][0], 0)
#         center = np.delete(center, index[0][0], 0)
#         # print(center.shape)
#         delta = np.diff(center, axis=0)
#         index = np.where(abs(delta) > 20)
#         # print(index[0])
#         # print(delta[index])


    # step = 10 #!!!!!!!!!!
    # delta_a = np.array([np.sum(delta[i:i + step,0]) for i in range(0, len(delta[:,0]), step)])
    # delta_b = np.array([np.sum(delta[i:i + step, 1]) for i in range(0, len(delta[:, 1]), step)])
    # delta = np.vstack((delta_a, delta_b)).T  # 更大步长的delta
    # center_a = np.array([np.average(center[i:i + step,0]) for i in range(0, len(center[:,0]), step)])
    # center_b = np.array([np.average(center[i:i + step, 1]) for i in range(0, len(center[:, 1]), step)])
    # center = np.vstack((center_a, center_b)).T  # 更大步长的center

    # for k in range(len(center[:,0])):
    #     center_c = center.copy()
    #     center_c = list(zip(center_c[:,0].tolist(),center_c[:,1].tolist()))
    #     center_c = np.delete(center_c,k,axis=0)
    #     # lof = LOF(center_c)
    #     # value = lof.local_outlier_factor(5, center[k])
    #     if value > 2:
    #         print(k, value,center[k])
    #     # print(k,'---',value, center[k])

    # if np.max(abs(delta)>11):
    #     c_mean = center[0,:]
    #     index2 = [0,]
    #     for k in range(1 ,len(center[:,0])):
    #         if np.linalg.norm(c_mean-center[k,:])<25: # 欧式距离
    #             index2.append(k)
    #             c_mean = center[index2,:].mean(axis=0)
    #     center = center[index2,:]
    #     delta = np.diff(center, axis=0)



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

    # print("Mann-Kendall:", Kendall_change_point_detection(dt))
    # print("Pettitt:", Pettitt_change_point_detection(dt))
    # print("Buishand U Test:", Buishand_U_change_point_detection(dt))
    # print("Standard Normal Homogeneity Test (SNHT):", SNHT_change_point_detection(dt))


 # ACF________________________________________-

    label.append(frame_name)
    step = 1 #!!!!!!!!!!
    x = center[:, 0]  # numpy array
    dx = x[step::1] - x[:-step:1]
    y = center[:, 1]
    dy = y[step::1] - y[:-step:1]
    if len(delta[:, 1]) > 10000:#
        dx = delta[:10000, 0]
        dy = delta[:10000, 1]
        x = center[:10000, 0]
        y = center[:10000, 1]
    # dr = np.sqrt(x ** 2 + y ** 2)
    dr = np.sqrt(dx ** 2 + dy ** 2)
    # dr=dx
    v_t = dr/ 111.0*0.04 # dr/ 111.0*0.04 #
# # method 1 --my function-------
    ac = autocorr(v_t)
    nan = np.isnan(ac)
    ac = ac[~nan]
    # # ac = smooth(ac)
    # # ac = ac[5:]
    # # ac /= ac[0]
    # ac = np.insert(ac,0,1.0)
    time = np.around(np.arange(0, (len(ac) + 1) * 1 / fps, 1 / fps), decimals=3)
    ax.scatter(time[0:len(ac)], ac / ac[0], c='', alpha=0.65,s=5, edgecolor=colors[j],label=frame_name )  # auto correlation
    # ax.scatter(time[0:len(ac)], ac / ac[0], c=colors[j],marker='+', s=3,label=frame_name + 'g', cmap='hsv')  # auto correlation
    ax.plot(time[0:len(ac)], ac / ac[0], alpha=0.65,lw=1, color=colors[j],label=frame_name )  # auto correlation


#画包络线
    # signal = ac / ac[0]
    # analytic_signal = hilbert(signal)
    # amplitude_envelope = np.abs(analytic_signal)
    # instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * 150)
    #
    # fig = plt.figure()
    # ax0 = fig.add_subplot(211)
    # ax0.plot(time[1:], signal, label='signal')
    # ax0.plot(time[1:], amplitude_envelope, label='envelope',lw=0.5)
    # ax0.set_xlabel("t")
    # ax0.legend()
    # ax1 = fig.add_subplot(212)
    # ax1.plot(time[2:], smooth(instantaneous_frequency)[:len(time[2:])]) #曲线的高度是对应时间尺度下出现的频率
    # ax1.set_xlabel("t")
    # # ax1.set_ylim(0.0, 120.0)

# # -------------提取频率
#     ac = autocorr(v_t)
#     nan = np.isnan(ac)
#     ac = ac[~nan]
#     ac_f = scipy.fftpack.fft(ac)
#     freqs = np.linspace(0.0, 150.0/2.0, int(len(ac) / 2.0))
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111)
#     ax.plot(freqs, 2.0 / len(ac) * np.abs(ac_f[:len(ac) // 2]),alpha=0.65,color=colors[j])
#     # plt.show()



    #
    # center_all[file_n[j] + 'x'] = np.sqrt(delta[:, 0]**2+delta[:,1]**2)  # center[:, 0]   #   np.sqrt(center[:, 0]**2+center[:,1]**2)  # delta[:,0]  #
    # center_all[file_n[j]+ 'y'] = center[:, 1]
    # delta_all[file_n[j] + 'x'] = delta[:18000, 0]
    # delta_all[file_n[j] + 'y'] = delta[:18000, 1]



# # ----------------------------------------------------------------------------------------------------------------------------------
# # # -------------d_all自相关--------------------
# plt.figure()
# time = np.around(np.arange(0, len(delta[:,1]) * 1 / fps*step, 1 / fps*step), decimals=2) # np.around(np.arange(0,len(delta[:,1]),1))  #
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(file_n)):  #  2,3):    #
#     plot_acf(delta_all[file_n[i]+'x'], lags=8000,alpha=1, marker='+', markersize=3, use_vlines=False, zero=True, ax=ax1,color=colors[i],title=' ') #
#     # plt.xscale('log')
#     label.append(filename1[i].split('_', 1)[0] + 'g')
#     # print(u'原始序列的ADF检验结果为：', ADF(center_all[filename[i]+'x']))
#     # print(u'原始序列的白噪声检验结果为：', acorr_ljungbox(center_all[filename[i]+'x'], lags=1))  # 返回统计量和p值
# plt.legend(label)
# plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# plt.show()


# # # -------------d_all偏自相关--------------------
# plt.figure()
# label = []
# time = np.around(np.arange(0, len(delta[:,1]) * 1 / fps*step, 1 / fps*step), decimals=2) # np.around(np.arange(0,len(delta[:,1]),1))  #
# ax1 = plt.subplot(111)
# for i in range(len(filename)):  #  2,3):    #
#     plot_acf(center_all[filename[i]+'x'], lags=time,alpha=1, marker='+', markersize=3, use_vlines=False, zero=True, ax=ax1,color=colors[i],title=' ')  # alpha=1,
#     # plt.xscale('log')
#     label.append(filename[i] + 'g')
# plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# plt.legend(label)
# plt.show()


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

# #----------auto correlation---------------------
ax.set_xlabel('$time~(s)$') #('lags')#
# ax.set_ylabel('$C_{\Delta \Theta}(t)$')
ax.set_ylabel('$C_{\Delta r}(t)$')#('$C_{\Delta X}(t)$')
# ax.set_ylabel('$C_{\Delta\Theta-\Delta\Theta}(t)$')

# # #----------Fourier------
# ax.set_xlabel('f(Hz)') #('lags')#
# # ax.set_ylabel('$fft$')
#

# ax.set_xscale('log')
leg = plt.legend(label,loc='upper right')
leg.get_frame().set_linewidth(0.0)
plt.show()






