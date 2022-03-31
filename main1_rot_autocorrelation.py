#  给出旋转的traj；顺势轴和角度，euler，四元数，angular displacement；主要用于计算PDF（方法二）
#   'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\' 是用来画rotational pdf的！！！！！

import tables as tb
import math
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import warnings
import sympy as sp
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
from matplotlib.ticker  import MultipleLocator
import pandas as pd



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
path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot(euler1)_acf_selected\\'
# 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rotacf\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！

filename = [name for name in os.listdir(path2)]
pdf_rot_dict = {}

fig = plt.figure()
ax = fig.add_subplot(111)
label = []
pdf_trans_dict = {}
center_all = {}
delta_all = {}
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))

step = 1  #30
label = []
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
        points = store.get('points').values[::step]  # timestep 1/5s
        store.close()

        N = len(center)
        max_time = N / fps  # seconds
        frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
        print(frame_name)



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

        # # -------- 单个点位移 --计算
        deltar = np.diff(points_reshape[:, 0, :], axis=0)  #------------单个点位移
        deltapr = np.sqrt(np.sum(deltar ** 2, axis=1))  # 每次旋转的球面距离(pixel)
        index = np.where(deltapr == 0)
        deltapr = np.delete(deltapr, index[0])

        d_theta = np.hstack((d_theta, deltatheta)) #全正theta
        d_axis = np.vstack((d_axis, deltaaxis))[1:,:]  # 去掉一开始那个[0,0,0]
        d_euler = np.vstack((d_euler, deltaeuler))[1:,:]  # 去掉一开始那个[0,0,0]
        d_pr = np.hstack((d_pr, deltapr))


        index = np.where(d_axis[:,2] <= 0)  # 正负theta（轴z+）
        d_new_theta = d_theta.copy()
        d_new_theta[index] = d_new_theta[index]*(-1)  # 把瞬时轴指向z=0以下的旋转角设为负
        d_new_axis = d_axis.copy()  # d_axis旋转轴指向z>0/<0都有，d_theta只有顺时针
        d_new_axis[index] = d_new_axis[index]*(-1)  # d_new_axis旋转轴全部指向z>0，d_new_theta有顺/逆时针

        d_quaternions = np.vstack((d_quaternions,quaternions)) # 四元数

    # ACF________________________________________-

    label.append(str(filename[j]) + 'g')
    # # method 1 --my function-------
    # if len(deltatheta) > 30000:  # 10000:#
    #     dx = np.cumsum(d_new_theta[:30000])  #  quaternions[:5000,3] # np.cumsum(deltaeuler[:,2]) #    
    # else:
    #     dx = np.cumsum(d_new_theta)  #  quaternions[:5000,3] # np.cumsum(deltaeuler[:,2]) #  
    dx = np.cumsum(deltaeuler[:,0]) # np.cumsum(d_new_theta)  
    
    # method 1-1
    v_t = dx
    ac = autocorr(v_t)
    nan = np.isnan(ac)
    ac = ac[~nan]
    # # ac = smooth(ac)
    # # ac = ac[5:]
    # # ac /= ac[0]
    # ac = np.insert(ac,0,1.0)
    time = np.around(np.arange(0, (len(ac) + 1) * 1 / 150, 1 / 150), decimals=3)
    ax.scatter(time[0:len(ac)], ac / ac[0], c='', alpha=0.65, s=5, edgecolor=colors[j], label=str(filename[j]) + 'g')  # auto correlation
    # ax.scatter(time[0:len(ac)], ac / ac[0], c=colors[j],marker='+', s=3,label=frame_name + 'g', cmap='hsv')  # auto correlation
    ax.plot(time[0:len(ac)], ac / ac[0], alpha=0.65, lw=1, color=colors[j],label=str(filename[j]) + 'g')  # auto correlation

    # # method 1-2
    # v_t = pd.Series(dx)
    # time = np.around(np.arange(0, (len(dx)) * 1 / 150, 1 / 150), decimals=3)
    # ax.plot(time, [v_t.autocorr(j) for j in range(len(v_t))], alpha=0.65, lw=1, color=colors[j],label=str(filename[j]) + 'g')






    # # method 2 --from statsmodels.tsa.stattools import acf
    # ac = np.cumsum(d_new_theta)  # quaternions[:,2] #  deltaeuler[:,1] #   
    # ac_n = acf(ac,lags=30000)
    # # plot_acf(ac, lags= 30000, alpha=0.05)
    # time = np.around(np.arange(0, (len(ac) + 1) * 1 / 150, 1 / 150), decimals=3)
    # ax.scatter(time[0:len(ac_n)], ac_n , c='', alpha=0.65, s=5, edgecolor=colors[j], label=str(filename[j]) + 'g')  # auto correlation
    # ax.plot(time[0:len(ac_n)], ac_n, alpha=0.65, lw=1, color=colors[j],label=str(filename[j]) + 'g')  # auto correlation




# #----------auto correlation---------------------

ax.set_xlabel('$time~(s)$') #('lags')#
# ax.set_ylabel('$C_{\Delta \Theta}(t)$')
ax.set_ylabel(r'$C_{\alpha}(t)$')
# ax.set_ylabel('$C_{q_4}(t)$')
# # ax.set_ylabel('$C_{\Delta\Theta-\Delta\Theta}(t)$')
# ax.yaxis.set_minor_locator(MultipleLocator(0.5))
# ax.tick_params(axis="x", direction="in")
# ax.tick_params(axis="y", direction="in")
# ax.tick_params(which='minor', direction='in')

# # #----------Fourier------
# ax.set_xlabel('f(Hz)') #('lags')#
# # ax.set_ylabel('$fft$')
#

ax.set_xscale('log')
# ax.set_yscale('log')

plt.axhline(y=0, c="r", ls="--", lw=1, alpha=0.3)
# leg = plt.legend(label[::])
# leg.get_frame().set_linewidth(0.0)
plt.show()
