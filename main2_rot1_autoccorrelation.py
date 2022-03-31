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
import sympy as sp

warnings.filterwarnings('ignore')
# TODO: CHANGE PARAMETERS HERE------------------
fps = 240.0
step = 1 #!!!!!!!!!!


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
path2 = 'G:\\ball_new\\3_analysis\\1\\' # autocorr\\' 
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
    matrix = store.get('matrix').values
    points = store.get('points').values # timestep 1/5s
    store.close()

    frame_name = filename1[j].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    # print(type(frame_name))

    delta = np.diff(center, axis=0)
    # print(center.shape)
    # print(np.min(delta),np.max(delta))
    index = np.where(abs(delta)>25)
    # print(index[0])

# rotational--------------------------------------------------------------------------------------------------------------
    points_reshape = np.reshape(points, (len(points), 6, 3))  # points 2维，points_reshape 3维

    if step != 1:
        # 重新算更改步长后的matrix-------------
        matrix = []
        for k in range(1, len(points)):
            points_co = points_reshape[k-1][0:3,:]
            points_ps_n = points_reshape[k][0:3,:]
            pi = sp.Matrix(points_co).T  # 转置
            pf = sp.Matrix(points_ps_n).T
            if pi.det() != 0:  # del()不为0
                rot = pf * (pi.inv())  # 旋转矩阵
                matrix.append(rot)


    matrix_reshape = np.reshape(np.array(matrix), (len(np.array(matrix)), 3, 3))  # reshape的矩阵
    points_1 = points_reshape[0][0]
    print(points_1)
    for k in range(1,len(points)-1):
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
    print(np.arccos(float(np.ndarray.trace(matrix_reshape[0]) - 1) / 2.))
    points_theta = np.nan_to_num(np.array([np.arccos(float(np.ndarray.trace(x) - 1) / 2.)+0.00001 for x in
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

    # ------------瞬时角+瞬时轴+欧拉角 +四元数--处理
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

    
    index = np.where(deltaaxis[:,2] <= 0)  # 正负theta（轴z+）
    d_new_theta = deltatheta.copy()
    d_new_theta[index] = d_new_theta[index]*(-1)  # 把瞬时轴指向z=0以下的旋转角设为负
    d_new_axis = deltaaxis.copy()  # d_axis旋转轴指向z>0/<0都有，d_theta只有顺时针
    d_new_axis[index] = d_new_axis[index]*(-1)  # d_new_axis旋转轴全部指向z>0，d_new_theta有顺/逆时针


 # ACF________________________________________-

    label.append(frame_name)
    if len(d_new_axis[:, 1]) > 10000:#
        dx = deltaeuler[:10000,0] # d_new_theta[:10000] 

    v_t = np.cumsum(dx) #dx # 
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

#         plt.figure()
#         # plt.plot(time[::50],smooth(corr)[:len(time):50],'.-', markerfacecolor='none',alpha=0.3)
#         plt.plot(time, corr[:len(time)], 'o',alpha=0.65, color=colors[i], label=frame_name + 'Hz')
#         plt.xscale('log')
#         plt.show()

# #----------auto correlation---------------------
ax.set_xlabel('$time~(s)$') #('lags')#
ax.set_ylabel('$C_{\Delta \\alpha }(t)$') # ('$C_{\Delta \Theta}(t)$')
# ax.set_ylabel('$C_{\Delta r}(t)$')#('$C_{\Delta X}(t)$')
# ax.set_ylabel('$C_{\Delta\Theta-\Delta\Theta}(t)$')

# # #----------Fourier------
# ax.set_xlabel('f(Hz)') #('lags')#
# # ax.set_ylabel('$fft$')
#

# ax.set_xscale('log')
leg = plt.legend(label,loc='upper right')
leg.get_frame().set_linewidth(0.0)
plt.show()






