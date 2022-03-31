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
    # theta_1 = store.get('theta').values
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


    x = center[:, 0] # numpy array
    dx = (x[step::step] - x[:-step:step])/(step/fps)
    y = center[:, 1]
    dy = (y[step::step] - y[:-step:step])/(step/fps)
    N = len(dy)

    traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'v_x': dx, 'v_y': dy})
    print(type(frame_name))

    x = center[:, 0]  # numpy array
    dx = x[step::1] - x[:-step:1]
    y = center[:, 1]
    dy = y[step::1] - y[:-step:1]
    # dr = np.sqrt(dx ** 2 + dy ** 2)
    dr = np.sqrt(x ** 2 + y ** 2)

    # # THETA = theta.reshape(len(center))
    # dtheta = THETA[step::1] - THETA[:-step:1]
    # index = []
    # for k in range(len(dtheta)):
    #     if dtheta[k] > 130:  # 处理周期行导致的大deltatheta
    #         dtheta[k] -= 180
    #     elif dtheta[k] < -130:
    #         dtheta[k] += 180
    #     if abs(dtheta[k]) > 60:  # 把明显由于识别错误产生的零星数据删掉
    #         index.append(k)
    # dtheta = np.delete(dtheta, index)
    # dr = np.delete(dr, index)

    v_t = dr/ 111.0*0.04  # 平动delta r  # (dx+dy)/2/ 111.0*0.04 
    # v_r = dtheta * math.pi / 180 # 转动delta theta

    time = np.arange(0, len(v_t) * step / fps, step / fps)

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

    f, Pxx_den = scipy.signal.periodogram(v_t, fps/step)  #todo 默认为功率谱密度，若要功率谱，加参数scaling='spectrum'


    print(Pxx_den.size)
    signal = smooth(Pxx_den)[:len(f)]
    # signal = Pxx_den
# 
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


    time = np.arange(0, len(v_t) * step / fps, step / fps)

# plt.xticks(np.arange(len(v_t)), time)
# ax.set_xlim((0.001, 1000))
# ax.set_ylim((1e-15, 1))

# ax.set_xscale('log')
# ax.xaxis.set_major_locator(ticker.LogLocator(10))
ax.set_xlabel('$f~(Hz)$')
ax.set_yscale('log')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')

# ax.set_ylabel('$S_{\Delta \Theta}(rad^2/Hz)$')
# ax.set_ylabel('$S_{\Delta x}~(m^2/Hz)$')
ax.set_ylabel('$S_{r}~(m^2/Hz)$')
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

