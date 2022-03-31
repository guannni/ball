# traj， fluctuation，球心PDF！！！ 
# fast motion ball (pingpangball)


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
from numpy import polyfit, poly1d


warnings.filterwarnings('ignore')
# TODO: CHANGE PARAMETERS HERE------------------
fps = 240.0
step = 1 


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
path2 = 'G:\\ball_new\\2_data\\frame\\20hz_240_copy\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！#


filename = [name for name in os.listdir(path2) ]
pdf_trans_dict = {}
center_all = {}
delta_all = {}
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for j in range(len(filename)):  #3,4):    #
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3) if name.endswith('.h5')]
    file_n = [path3 + name + '.h5' for name in filename1]
    print(filename1, file_n)

    d_all = []  # 合在一起的dx和dy

    for i in range(len(file_n)):  # 2,3):    #   # TODO 在len(file_n)):  #这里更改文件，把同意条件下dx dy合在一起计算pdf
        store = pd.HDFStore(file_n[i], mode='r')
        print(store.keys())
        center = store.get('center').values  # numpy array
        store.close()


        frame_name = filename1[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
        print(frame_name)

        delta = np.diff(center, axis=0)


        delta_a = np.array([np.sum(delta[i:i + step,0]) for i in range(0, len(delta[:,0]), step)])
        delta_b = np.array([np.sum(delta[i:i + step, 1]) for i in range(0, len(delta[:, 1]), step)])
        delta = np.vstack((delta_a, delta_b)).T  # 更大步长的delta
        center_a = np.array([np.average(center[i:i + step,0]) for i in range(0, len(center[:,0]), step)])
        center_b = np.array([np.average(center[i:i + step, 1]) for i in range(0, len(center[:, 1]), step)])
        center = np.vstack((center_a, center_b)).T  # 更大步长的center



        N = len(center[:,0])
        max_time = N / fps * step # seconds
        time = np.linspace(0, N-1, N) *step/fps # (0, max_time, N)
        dt = max_time/N

        # # # dx + dy----
        # d_all = np.hstack((d_all,delta[:,0]))
        # d_all = np.hstack((d_all,delta[:,1]))
        # dr----
        # d_all = np.hstack((d_all,np.sqrt(delta[:,0]**2+delta[:,1]**2)))
        # p(r,t)--------待改
        center_a = center[:,0]/time**0.6
        center_b = center[:, 1]/time**0.6
        center = np.vstack((center_a, center_b)).T  # 更大步长的center
        d_all = np.hstack((d_all,np.sqrt(center[:,0]**2+center[:,1]**2)))


        # -flow/////////////dx + dy----
        # d_all = np.hstack((d_all,np.diff(delta[:,0])))
        # d_all = np.hstack((d_all,np.diff(delta[:,1])))
        # # # -flow/////////////dr----
        # d_all = np.hstack((d_all,np.diff(np.sqrt(delta[:,0]**2+delta[:,1]**2))))


        print(d_all.shape)


    
    # # -------每个文件traj pic/相图--------------------
    #     # #--1
    #     # traj = pd.DataFrame({'t': time, 'x': center[:,0], 'y': center[:,1]}) # delta[:,1]})#
    #     # # print(traj.head())
    #     # ax = traj.plot(x='x', y='y', alpha=0.6, legend=False, title='trajectory')
        #--2
        fig, ax = plt.subplots(nrows=1, figsize=(6, 6))
        ax = plt.subplot(111)
        plt.scatter(np.diff(np.diff(np.sqrt(center[:,1]**2+center[:,0]**2))), np.diff(np.sqrt(center[:-1,1]**2+center[:-1,0]**2)),s=1) # ddr-dr
        # coeff = polyfit(np.diff(np.diff(np.sqrt(center[:,1]**2+center[:,0]**2)))[:10000], np.diff(np.sqrt(center[:-1,1]**2+center[:-1,0]**2))[:10000], 1)
        # plt.scatter(np.diff(center[:-1,0]),np.diff(np.diff(center[:,0])),s=1) # ddx-dx

        # plt.scatter(np.sqrt(center[:-1,1]**2+center[:-1,0]**2), np.sqrt(delta[:,1]**2+delta[:,0]**2),s=1) 
        # plt.scatter(center[:-1,1], delta[:,1],s=1)
        # plt.scatter(center[100000:150000:1,0], center[100000:150000:1,1],s=1)
        # plt.plot(center[100000:105000:1,0], center[100000:105000:1,1],lw=0.3)

        
        print(coeff)

        # ax.set_xlim(0, 1180)  # (traj['x'].min()-10, traj['x'].max()+10)
        # ax.set_ylim(0, 1180)  # (traj['y'].min()-10, traj['y'].max()+10)
        ax.set_xlabel('y(pixel)')
        ax.set_ylabel('dy(pixel)')
        ax.plot()
        # plt.text(1.5, 1, '1 pixel=0.54 mm', fontsize=10, horizontalalignment='left', verticalalignment='bottom')
        plt.show()
        fig = ax.get_figure()
        del ax, fig



    # # # -------每个文件fluctuation--------------------
    #     fig, (ax0, ax1) = plt.subplots(nrows=2, figsize=(12, 6))
    #     ax0 = plt.subplot(211)
    #     # plt.plot(time, center[:, 0], lw=0.5, marker='o', markersize=0.7)  # Displacement Position pixel  # mm / 480.0 * 260
    #     plt.scatter(time[:len(delta[:,0])], np.sqrt(delta[:,0]**2+delta[:,1]**2),s=3) #center[:, 0], s=3) # Displacement Position pixel # mm / 480.0 * 260
    
    #     ax0.set_xlabel('time (s)')
    #     ax0.set_ylabel('$\Delta r (pixel)$')
    #     ax0.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    #     ax0.set_xlim(time[:len(delta[:,0])][0], time[:len(delta[:,0])][-1])
    #     plt.title('Position' + '[' + str(frame_name) + ']', fontsize=10)  #  Displacement  Position
    
    #     ax1 = plt.subplot(212)
    #     plt.scatter(time, center[:, 1], s=3)  # pixel
    #     ax1.set_xlabel('time (s)')
    #     ax1.set_ylabel('$y (pixel)$')
    #     ax1.set_xlim(time[0], time[-2])
    #     ax1.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.95')  # 由每个x主坐标出发对x主坐标画垂直于x轴的线段
    #     plt.show()

    # center_all[filename[j] + 'x'] = np.sqrt(delta[:, 0]**2+delta[:,1]**2)  # center[:, 0]   #   np.sqrt(center[:, 0]**2+center[:,1]**2)  # delta[:,0]  #
    # center_all[filename[j] + 'y'] = center[:, 1]
    # delta_all[filename[j] + 'x'] = delta[:, 0]
    # delta_all[filename[j] + 'y'] = delta[:, 1]



# ------计算所有文件PDF--------------------
#     print(max(d_all),-min(d_all))
#     weights_r = np.ones_like(d_all) / float(len(d_all))
#     # cuts_m = int(max(-min(d_all), max(d_all)) / 0.5 // 2) + 0.75
#     # cuts = np.arange(-cuts_m, cuts_m + 0.5, 0.5)
#     cuts_m = int(max(-min(d_all), max(d_all)) / 0.5 // 2) + 0.5
#     cuts = np.arange(-cuts_m, cuts_m, 1)
#
#     print(cuts)
#
#     au, bu, cu = plt.hist(d_all, cuts, histtype='bar', facecolor='yellowgreen',
#                           weights=weights_r, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
#     pdf_trans_dict[filename[j]+'y'] = au
#     bu = (bu[:-1] + bu[1:]) / 2.
#     pdf_trans_dict[filename[j]+'x'] = bu  # 存入dict
#     print(filename[j]+'x')

    AU, BU, CU = plt.hist(d_all,np.arange(-15.25,15,0.5)*100, histtype='bar', facecolor='blue', alpha=0.75, rwidth=0.2)  # , density=True)
    pdf_trans_dict[filename[j] + 'y'] = AU / len(d_all)
    BU = (BU[:-1] + BU[1:]) / 2.
    pdf_trans_dict[filename[j] + 'x'] = BU  # 存入dict




# # pdf----------------------------------------------------------------------------------------------------------------
#  ----------------画pdf------------------
ys = [i +  (i ) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

fig = plt.figure()
ax1 = fig.add_subplot(111)
label = []
for i in range(len(filename)):  #  2,3):    #
    plt.plot(pdf_trans_dict[filename[i]+'x'] , pdf_trans_dict[filename[i]+'y'],'o-',alpha=0.5,color=colors[i])#, cmap='hsv')
    # ax1.scatter(pdf_trans_dict[filename[i] + 'x'], pdf_trans_dict[filename[i] + 'y'], alpha=0.75,c='', s=25, edgecolor=colors[i])  # , label=label[i])

    label.append(filename[i])
ax1.legend(label)

#
# ########## Translatioanl fitting
# for i in range(len(filename)):  #  2,3):    #
#     middle = np.where(pdf_trans_dict[filename[i]+'x']  == find_nearest(pdf_trans_dict[filename[i]+'x'], 0))[0][0]  # pdf 0 的点
#     x2 = pdf_trans_dict[filename[i]+'x'][:middle+1]  # 越过中间的几个点
#     y2 = pdf_trans_dict[filename[i]+'y'][:middle+1]
#     x1 = -np.flipud(pdf_trans_dict[filename[i]+'x'][middle:])
#     y1= np.flipud(pdf_trans_dict[filename[i]+'y'][middle:])

#     x = np.array((x1+x2)/2.0)[5:]
#     y = np.array((y1+y2)/2.0)[5:]
#     popt, pcov = curve_fit(func, x, y, maxfev=1000)
#     y11 = [func(p, popt[0],popt[1],popt[2]) for p in x]
#     print(filename[i])
#     print(popt)
#     print('---')
#     plt.plot(-np.flipud(x),np.flipud(y11),color=colors[i], alpha=0.5)
#     # popt, pcov = curve_fit(func, x2, y2, maxfev=1000)
#     # y21 = [func(p, popt[0],popt[1],popt[2]) for p in x2]
#     # plt.plot(-np.flipud(x1),np.flipud(y11),x2,y21,color=colors[i])
#     # print(popt)
#     # # print(pcov)
#     # print('--')
#     # #--------计算拟合优度 r^2---
#     # # residual sum of squares
#     # ss_res1 = np.sum((y1 - y11) ** 2)
#     # ss_res2 = np.sum((y2 - y21) ** 2)
#     # # total sum of squares
#     # ss_tot1 = np.sum((y1 - np.mean(y1)) ** 2)
#     # ss_tot2 = np.sum((y2 - np.mean(y2)) ** 2)
#     # # r-squared
#     # r21 = 1 - (ss_res1 / ss_tot1)
#     # r22 = 1 - (ss_res2 / ss_tot2)
#     # print(r21,'---',r22)

leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)
ax1.set_title('Translational PDF' + ' [20Hz] ', fontsize=10)
ax1.set_xlabel('$\Delta r(pixel)$')
ax1.set_ylabel('$P_{\Delta r}$')
plt.yscale('log')
ax1.set_ylim(0.0001, 100)
plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
plt.axvline(x=0, c="r", ls="--", lw=1, alpha=0.3)

plt.show()



# # ----------------------------------------------------------------------------------------------------------------------------------
# # # -------------d_all自相关--------------------
# plt.figure()
# time = np.around(np.arange(0, len(delta[:,1]) * 1 / 150*step, 1 / 150*step), decimals=2) # np.around(np.arange(0,len(delta[:,1]),1))  #
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  #  2,3):    #
#     plot_acf(center_all[filename[i]+'x'], lags=time,alpha=1, marker='+', markersize=3, use_vlines=False, zero=True, ax=ax1,color=colors[i],title=' ') #
#     plt.xscale('log')
#     label.append(filename[i] + 'g')
#     # print(u'原始序列的ADF检验结果为：', ADF(center_all[filename[i]+'x']))
#     # print(u'原始序列的白噪声检验结果为：', acorr_ljungbox(center_all[filename[i]+'x'], lags=1))  # 返回统计量和p值
# plt.legend(label)
# plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# plt.show()
#
# # # -------------d_all偏自相关--------------------
# plt.figure()
# label = []
# time = np.around(np.arange(0, len(delta[:,1]) * 1 / 150*step, 1 / 150*step), decimals=2) # np.around(np.arange(0,len(delta[:,1]),1))  #
# ax1 = plt.subplot(111)
# for i in range(len(filename)):  #  2,3):    #
#     plot_acf(center_all[filename[i]+'x'], lags=time,alpha=1, marker='+', markersize=3, use_vlines=False, zero=True, ax=ax1,color=colors[i],title=' ')  # alpha=1,
#     # plt.xscale('log')
#     label.append(filename[i] + 'g')
# plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# plt.legend(label)
# plt.show()

