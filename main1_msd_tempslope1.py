# 读取hdf文件(msd)
# 球心平动 'D:\\guan2019\\2_ball\\3_ananlysis\\trans\\msd\\60Hz_tt\\selected\\'


import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
from scipy.ndimage import filters

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

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def abline(slope, intercept):
    axes = plt.gca()
    x_vals = np.array(np.logspace(-2, 2, 5))
    y_vals = intercept + slope * x_vals
    plt.plot(np.logspace(-2, 2, 5), np.logspace(-2, 2, 5), '--')

def smooth(x, window_len=5, window='flat'):
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



# path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_trans_msd_selected\\' # translation
# path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(axisangle)_msd_selected\\'  # angleaxis
# path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(euler3)_msd_selected\\'  # euler
path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(qua1)_msd_selected\\'  # quaternions

#'D:\\guan2019\\2_ball\\3_ananlysis\\trans\\msd\\60tt\\selected\\'  # 平动
# path2 = 'D:\\guan2019\\2_ball\\3_ananlysis\\rot\\msd\\theta\\60Hz\\'  # rotate theta
filename = [name for name in os.listdir(path2)]
fig = plt.figure()
ax = fig.add_subplot(111)
label = []
ys = [m + (m) ** 2 for m in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

for j in range(len(filename)):  #3,4):    #
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3) if name.endswith('.h5')] # 只取.h5文件
    file_n = [path3 + str(name) + '.h5' for name in filename1]
    print(filename1, file_n)

    msd_all = []
    tau = []
    lt = 0



    for i in range(len(file_n)):  #0,1):#
        store = pd.HDFStore(file_n[i], mode='r')
        print(store.keys())
        MSD_key = store.keys()[0]
        MSD = store.get(MSD_key).values[1:, 0]*26*26/48/48  # filters.gaussian_filter1d(store.get(MSD_key).values[:, 0], 3)  #
        # MSD = store.get(MSD_key).values[1:, 0]
        TAU = store.get(MSD_key).values[1:, 1]
        store.close()
        if i == 0 :
            tau = TAU
        print(len(TAU),len(tau))
        if len(TAU)<len(tau):
            tau = TAU
        print(len(tau))


        if i == 0 :
            msd_all = MSD
            lt = len(MSD)
        elif i == 1 :
            msd_1 = msd_all
            print(msd_1.shape[0])
            msd_all = np.zeros((i + 1, max(msd_1.shape[0],len(MSD))))
            msd_all[0,:msd_1.shape[0]] = msd_1
            msd_all[1,:len(MSD)] = MSD
            lt = min(lt,len(MSD))

        elif i > 1:
            msd_1 = msd_all
            msd_all = np.zeros((i+1,max(msd_1.shape[1],len(MSD))))
            msd_all[0:i,:msd_1.shape[1]] = msd_1
            msd_all[i,:len(MSD)] = MSD
            lt = min(lt, len(MSD))

    if len(file_n)>1:
        msd_m = msd_all.mean(axis=0)[:lt]
    else:
        msd_m = msd_all

    label.append(str(filename[j]) + 'g')

    step = 1
    slope = (np.log(msd_m[1:]) - np.log(msd_m[:-1])) / (np.log(tau[1:]) - np.log(tau[:-1]))  # np.power(10, np.gradient(np.log(MSD)))
    N = len(slope)
    max_time = N / 150 * step
    time = np.linspace(0, max_time, N)
    slope = smooth(np.array(slope))[-len(time):]

    a = np.logspace(-3, 2, 50)
    a_index = []
    time_new = []
    slope_new = []
    for i in a:
        ind = find_nearest(time, i)
        a_index.append(ind)
        time_new.append(time[ind])
        slope_new.append(slope[ind])

    plt.plot(time_new, slope_new, 'o-', markerfacecolor='none', alpha=0.75, color=colors[j])
    # plt.plot(time_new[1:], smooth(slope_new)[-(len(time_new)-1):], alpha=0.75, color=colors[j])

    # print(len(tau), len(slope))
    # plt.scatter(tau[1:]-1/300.0, smooth(slope)[-(len(tau)-1):], alpha=0.5, color='', edgecolors=colors[j], cmap='hsv')
    # plt.plot(tau[1::50]-1/300.0, smooth(slope)[-(len(tau)-1)::50], 'o-', alpha=0.5, color=colors[j],markersize=5)  # , markerfacecolor='none',alpha=1)
    # plt.plot(tau[1:]-1/300.0,slope,'o-', markerfacecolor='none',alpha=0.5, color=colors[j], markersize=1)
    # plt.scatter(tau[1:]-1/300.0, slope, alpha=0.5, color='', edgecolors=colors[j], cmap='hsv')

# plt.ylim(-1,2)
# plt.xlim(0.001,60)
plt.xscale('log')

# ax.set_title('Local Slope of Rotational MSD [60Hz]', fontsize=10)
ax.set_title('Local Slope of Translational MSD [60Hz]', fontsize=10)
ax.set_xlabel(r'$time~(s)$')
ax.set_ylabel('$\u03B1 $')
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)

# plt.axvline(x=.1, c="r", ls="--", lw=0.3, alpha=0.8)
# plt.axvline(x=.2, c="r", ls="--", lw=0.3, alpha=0.8)
plt.axhline(y=0, c="r", ls="--", lw=0.3, alpha=0.8)
plt.axhline(y=1, c="r", ls="--", lw=0.3, alpha=0.8)
# plt.annotate(r'$1.75$',c='r', xy=(2, 1), xytext=(0.0015,1.6), xycoords='data')


plt.show()

print(label)