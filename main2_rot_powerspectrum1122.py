# 读取hdf文件，计算v
import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import math
import os.path
import scipy
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import seaborn as sns
from scipy.stats.stats import pearsonr
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.ticker as ticker
from scipy.fftpack import fft,ifft
from scipy.signal import hilbert, chirp
from matplotlib.ticker  import MultipleLocator
from matplotlib.ticker import FuncFormatter
import sympy as sp

# TODO: CHANGE PARAMETERS HERE------------------
fps = 240.0
step = 1
FRE = 20  # todo eachF [50,85,5]没有80，手动排着输
# ACC = 5 # todo eachA [3,5,0.5]

# -----------------------------------------------
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


def smooth(x, window_len=35, window='flat'):
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


path2 = 'G:\\ball_new\\3_analysis\\1\\'#spectrum_trans\\' 

filename = [os.path.splitext(name)[0] for name in os.listdir(path2) if name.endswith('.h5')]
file_n = [path2 + name + '.h5' for name in filename]
print(filename, file_n)



fig = plt.figure()
ax = fig.add_subplot(111)
label = []
ys = [i + (i) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
for i in range(len(file_n)):#
    store = pd.HDFStore(file_n[i], mode='r')
    print(store.keys())
    center_1 = store.get('center').values  # numpy array
    matrix = store.get('matrix').values
    points = store.get('points').values[::step]  # timestep 1/5s
    store.close()

    # todo : 改变delta间隔------------------
    # 这里改了要在最后图片标题maxtime改一下！
    center = center_1[::step]
    # theta = theta_1[::step]
  
    # -------------------------------------
    N = len(center[:,0])

    max_time = N / fps * step  # seconds
    frame_name = filename[i].split('_', 1)[0]  # 频率 为.h5文件的key，后面多组数据作图用key来挑选！！！
    # if len(frame_name)>3:
    #     frame_name = frame_name[1:]

# # translation------------------------------------------------------------------
#     x = center[:, 0] # numpy array
#     dx = (x[step::step] - x[:-step:step])/(step/fps)
#     y = center[:, 1]
#     dy = (y[step::step] - y[:-step:step])/(step/fps)
#     N = len(dy)

#     traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'v_x': dx, 'v_y': dy})
#     print(type(frame_name))

#     x = center[:, 0]  # numpy array
#     dx = x[step::1] - x[:-step:1]
#     y = center[:, 1]
#     dy = y[step::1] - y[:-step:1]
#     dr = np.sqrt(dx ** 2 + dy ** 2)

# rotational--------------------------------------------------------------------------------------------------------------
    points_reshape = np.reshape(points, (len(points), 6, 3))  # points 2维，points_reshape 3维

    matrix_reshape = np.reshape(np.array(matrix), (len(np.array(matrix)), 3, 3))  # reshape的矩阵
    points_1 = points_reshape[0][0]
    print(points_1)
    # for k in range(1,len(points)-1):
    #     # print(np.linalg.norm(points_reshape[k][0]-points_1))
    #     # print(points_1)
    #     if np.linalg.norm(points_reshape[k][0] - points_1) > 10:
    #         # print(points_reshape[k][0],k)
    #         points_reshape[k] = points_reshape[k - 1]
    #         matrix_reshape[k - 1] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #         matrix_reshape[k] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    #     points_1 = points_reshape[k][0]


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
    deltatheta = points_theta
    deltaeuler = deltaeuler 

    all = np.vstack((deltatheta, points_axis[:, 0], points_axis[:, 1],
                        points_axis[:, 2], deltaeuler[:, 0], deltaeuler[:, 1], deltaeuler[:, 2], quaternions[:, 0],
                        quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]))  # theta[0,:],axis[1:4,:],euler[4:7,:],quaterions[7:11,:]


    # index = np.where(deltatheta == 0)
    # all = np.delete(all, index[0],axis=1)
    # deltaaxis = np.delete(points_axis, index[0],axis=0)  # 瞬时轴
    # index = np.where(np.nan_to_num(deltaaxis[:,0] == 0))
    # all = np.delete(all, index[0],axis=1)
    #
    # index = []
    # for k in range(1,len(all[0,:])):
    #     if (all[:,k]==all[:,k-1]).all():
    #         index.append(k)
    #
    # all = np.delete(all, index,axis=1)

    deltatheta = all[0,:].T  # 每次旋转的角度 (rad)
    deltaaxis = all[1:4,:].T  # 瞬时轴
    deltaeuler = all[4:7,:].T # euler (rad)
    quaternions = all[7:11,:].T  # quaternions
    rot_energy = 0.1*0.5*9.028*deltatheta**2 # uJ
    deltatheta = deltatheta  # 每次旋转的角度 (rad)

    index = np.where(deltaaxis[:,2] <= 0)  # 正负theta（轴z+）
    d_new_theta = deltatheta.copy()
    d_new_theta[index] = d_new_theta[index]*(-1)  # 把瞬时轴指向z=0以下的旋转角设为负
    d_new_axis = deltaaxis.copy()  # d_axis旋转轴指向z>0/<0都有，d_theta只有顺时针
    d_new_axis[index] = d_new_axis[index]*(-1)  # d_new_axis旋转轴全部指向z>0，d_new_theta有顺/逆时针



    # # # -------- 单个点位移 --计算
    # deltar = np.diff(points_reshape[:, 0, :], axis=0)  #------------单个点位移
    # deltapr = np.sqrt(np.sum(deltar ** 2, axis=1))  # 每次旋转的球面距离(pixel)
    # index = np.where(deltapr == 0)
    # deltapr = np.delete(deltapr, index[0])

    # d_theta = np.hstack((d_theta, deltatheta)) #全正theta
    # d_rot_energy = np.hstack((d_rot_energy, rot_energy)) # rot energy
    # d_axis = np.vstack((d_axis, deltaaxis))[1:,:]  # 去掉一开始那个[0,0,0]
    # d_euler = np.vstack((d_euler, deltaeuler))[1:,:]  # 去掉一开始那个[0,0,0]
    # d_pr = np.hstack((d_pr, deltapr))

    # index = np.where(d_axis[:,2] <= 0)  # 正负theta（轴z+）
    # d_new_theta = d_theta.copy()
    # # d_new_theta = np.hstack(([0], np.diff(d_new_theta))) # -flow\\\\\\\\\\\\\\\\\\\\\
    # d_new_theta[index] = d_new_theta[index]*(-1)  # 把瞬时轴指向z=0以下的旋转角设为负
    # d_new_axis = d_axis.copy()  # d_axis旋转轴指向z>0/<0都有，d_theta只有顺时针
    # d_new_axis[index] = d_new_axis[index]*(-1)  # d_new_axis旋转轴全部指向z>0，d_new_theta有顺/逆时针

    # d_quaternions = np.vstack((d_quaternions,quaternions)) # 四元数




    # v_t = (dx+dy)/2/ 111.0*0.04  # 平动delta r
    v_r = deltaeuler[:,0]# 转动delta theta  # d_new_theta  # deltaeuler[:10000,2]  # np.cumsum(deltaeuler[:,2] )
    # v_r =np.cumsum(deltaeuler[2,:])   # 转动delta euler# np.cumsum(deltaeuler[2,:]) 

    time = np.arange(0, len(v_r) * step / fps, step / fps)

    # ps = np.abs(np.fft.fft(v_t)) ** 2
    # time_step = step/fps
    # freqs = np.fft.fftfreq(v_t.size, time_step)
    # idx = np.argsort(freqs)
    # plt.plot(freqs[idx], ps[idx])

    # rate = 1/(step/fps)
    # p = np.abs(np.fft.rfft(v_t))**2/(2*math.pi*len(v_t) * step / fps)
    # N = len(p)
    # p = p[0:round(N/2)]
    # fr = np.linspace(0,rate/2,N/2)
    # plt.plot(fr,abs(p)**2)

    f, Pxx_den = scipy.signal.periodogram(v_r, fps/step)  #todo 默认为功率谱密度，若要功率谱，加参数scaling='spectrum'


    print(Pxx_den.size)
    signal = smooth(Pxx_den)[:len(f)]
    # signal = Pxx_den

    plt.plot(f, signal,alpha=0.7,color=colors[i])
    # plt.scatter(f, signal,alpha=0.2,s=3,color=colors[i])
    # plt.scatter(f, Pxx_den,c='', alpha=0.35,s=5, edgecolor=colors[i])
    # plt.plot(f, Pxx_den, alpha=0.25,lw=0.5, color=colors[i])

    # plt.show()

# # 画包络线
#     analytic_signal = hilbert(signal)
#     amplitude_envelope = np.abs(analytic_signal)
#     instantaneous_phase = np.unwrap(np.angle(analytic_signal))
#     instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0 * np.pi) * 150)
    
#     fig = plt.figure()
#     ax0 = fig.add_subplot(211)
#     ax0.plot(f, signal, label='signal')
#     ax0.plot(f, amplitude_envelope, label='envelope',lw=0.5)
#     ax0.set_xlabel("f")
#     ax0.legend()
#     ax1 = fig.add_subplot(212)
#     ax1.plot(f[1:], instantaneous_frequency)
#     ax1.set_xlabel("f")
#     # ax1.set_ylim(0.0, 120.0)

    label.append(frame_name)


    time = np.arange(0, len(v_r) * step / fps, step / fps)

# plt.xticks(np.arange(len(v_t)), time)
# ax.set_xlim((0.001, 1000))
# ax.set_ylim((1e-15, 1))

ax.set_xscale('log')
# ax.xaxis.set_major_locator(ticker.LogLocator(10))
ax.set_xlabel('$f~(Hz)$')
ax.set_yscale('log')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')

ax.set_ylabel('$S_{\Delta \\alpha}(rad^2/Hz)$') # ('$S_{\Theta}(rad^2/Hz)$')
# ax.set_ylabel('$S_{\Delta x}~(m^2/Hz)$')
ax.set_xlim(1/fps, 130)

# plt.title('Translational Power Spectrum Density [0.6mm]')#, step=%s]' % str(step))
# plt.title('Rotational Power Spectrum Density [0.6mm]')#, step=%s]' % str(step))

# label = ['1','2','3']
# plt.legend(label)
leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)

#
# # 60Hz translational
# plt.plot(np.logspace(1,2,5), np.logspace(-9,-7.5, 5), 'r--')   # k=-2
# plt.annotate(r'$k=1.5$', xy=(2,4e-9), xytext=(2,4e-9), xycoords='data',color='r')


plt.show()

