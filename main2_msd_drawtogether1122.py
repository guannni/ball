# 读取hdf文件(msd)
# 球心平动 'G:\\ball_new\\3_analysis\\msd\\trans\\'

# import tables as tb
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import math
import matplotlib.cm as cm
from scipy.ndimage import filters
from matplotlib.ticker  import MultipleLocator
from matplotlib.ticker import FuncFormatter

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



path2 = 'G:\\ball_new\\3_analysis\\msd\\trans\\20hz_selected\\' # translation
# path2 = 'G:\\ball_new\\3_analysis\\msd\\rot_axisangle\\20hz\\'  # angleaxis
# path2 = 'G:\\ball_new\\3_analysis\\msd\\rot_euler3\\20hz_selected\\'  # euler
# path2 = 'G:\\ball_new\\3_analysis\\msd\\rot_qua4\\20hz\\'  # quaternions

#'D:\\guan2019\\2_ball\\3_ananlysis\\trans\\msd\\60tt\\selected\\'  # 平动
# path2 = 'D:\\guan2019\\2_ball\\3_ananlysis\\rot\\msd\\theta\\60Hz\\'  # rotate theta

filename = [name for name in os.listdir(path2)]
fig = plt.figure()
ax = fig.add_subplot(111)
label = []
ys = [m + (m) ** 2 for m in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
marker = ['o', 'v', 'D','^','s', 'h', '2', 'p', '*',  '+', 'x']

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
        # MSD = store.get(MSD_key).values[1:, 0]*26*26/48/48  # translational
        MSD = store.get(MSD_key).values[1:, 0]  # rotational
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

    label.append(str(filename[j]))

    a = np.logspace(-3,3,1000)
    a_index = []
    TAU_new = []
    MSD_new = []
    for i in a:
        ind = find_nearest(TAU, i)
        a_index.append(ind)
        TAU_new.append(TAU[ind])
        MSD_new.append(MSD[ind])

    # MSD_new = np.array(MSD_new)/180.0/180.0*math.pi*math.pi
    plt.plot(TAU_new, MSD_new,'o', markerfacecolor='none', alpha=0.75, color=colors[j])
    # ax.scatter(TAU_new, MSD_new, alpha=0.75,marker=marker[j], c='', s=25, edgecolor=colors[j])#, label=label[i])

    # plt.plot(TAU_new, MSD_new, 'o', markerfacecolor='none', alpha=0.75, color=colors[j])
    # plt.scatter(tau, msd_m,alpha=0.5, color='',edgecolors=colors[j],marker='d', cmap='hsv')
    # plt.plot(TAU, MSD, alpha=0.6,lw=3)



plt.ylim(0.001,100000000)
plt.xlim(0.001,1000)
plt.xscale('log')
plt.yscale('log')
ax.tick_params(axis="x", direction="in")
ax.tick_params(axis="y", direction="in")
ax.tick_params(which='minor', direction='in')

#
ax.set_title('Translational MSD [20Hz]', fontsize=10)
ax.set_ylabel(r'$\langle\Delta R^2 \rangle ~(pixel^2)$')
# ax.set_title('Rotational MSD [20Hz]', fontsize=10)
# ax.set_ylabel('$<\Delta \Theta ^2>(degree^2)$')
# ax.set_ylabel('$<\Delta \Theta ^2>(rad^2)$')
# ax.set_ylabel(r'$<\Delta \gamma ^2>(rad^2)$')
# ax.set_ylabel(r'$<Q_1 ^2>$')

ax.set_xlabel('$time~(s)$')

# label = ['mode1','mode2','mode3']#,'mode4']
# label = ['1','2','3']#,'mode4']


leg = plt.legend(label)
leg.get_frame().set_linewidth(0.0)

# plt.plot(np.logspace(0,1.5, 5), np.logspace(1.5,3, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(1.6,389), xycoords='data')

# # rotatioanl
# plt.plot(np.logspace(-0.5,0.5, 5), np.logspace(2.5,3.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.23,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.4, 5), np.logspace(-0.5,1.5, 5), 'b--')  # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.1,2), xycoords='data')

# # translational
# plt.plot(np.logspace(0, 1, 5), np.logspace(1.5,3.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(2,1800), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.4, 5), np.logspace(-0.3,0.8, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.1,5), xycoords='data')

# # 55 translational
# plt.plot(np.logspace(0, 1, 5), np.logspace(-1,1, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(4.8,0.5), xycoords='data')
# plt.plot(np.logspace(-1,1, 5), np.logspace(-0.1,1.8, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.1,5), xycoords='data')
# # plt.plot(np.logspace(-1.5,-0.5, 5), np.logspace(-2.3,-1.3, 5), 'b--')  # k=1
# # plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.1,0.01), xycoords='data')

# # 55 rotational
# plt.plot(np.logspace(-1, 1, 5), np.logspace(-3,1, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(4.8,0.5), xycoords='data')
# plt.plot(np.logspace(0.4,1.6, 5), np.logspace(4,5.2, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(2.3,50000), xycoords='data')
#
# # 70 rotational
# plt.plot(np.logspace(-2,-0.2, 5), np.logspace(-2,1.6, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(0.17, 1), xytext=(0.17,1.15), xycoords='data')
# plt.plot(np.logspace(0.6,1.6, 5), np.logspace(3.7,4.7, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(2.3,15900), xycoords='data')

# # # 70 translational
# plt.plot(np.logspace(-1,0, 5), np.logspace(-0.5,0.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(0.17, 1), xytext=(0.6,1.1), xycoords='data')
# plt.plot(np.logspace(0.5,1.5, 5), np.logspace(1,2.5, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1.5$', xy=(2, 1), xytext=(8,30), xycoords='data')

#################                   TLC----------
# # step=1 rotatioanl
# plt.plot(np.logspace(0.1,1.1, 5), np.logspace(2.7,3.7, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.8,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.6, 5), np.logspace(-0.5,1.5, 5), 'b--')  # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.01,2), xycoords='data')

# # step=1 translational
# plt.plot(np.logspace(0, 1, 5), np.logspace(1.5,3.5, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(2,1800), xycoords='data')
# plt.plot(np.logspace(-1.1,-0.1, 5), np.logspace(-0.8,0.2, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.39,0.56), xycoords='data')
################################################################################

####################                 60Hz----------------
# # step=1 translational
# plt.plot(np.logspace(1,1.5, 5), np.logspace(1,2, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(28,7.5), xycoords='data')
# plt.plot(np.logspace(-1.1,-0.1, 5), np.logspace(0,1, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.09,3.5), xycoords='data')

# step=1 rotatioanl
# plt.plot(np.logspace(0.1,1.1, 5), np.logspace(2.6,3.6, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.8,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.6, 5), np.logspace(-0.5,1.5, 5), 'b--')  # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.01,2), xycoords='data')

# ####################                 5g----------------
# # # step=1 translational
# plt.plot(np.logspace(0,1, 5), np.logspace(-0.8,1.2, 5), 'b--')   # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(1.8,0.23), xycoords='data')
# plt.plot(np.logspace(-1.1,-0.1, 5), np.logspace(0,1, 5), 'b--')  # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(0.09,3.5), xycoords='data')

# step=1 rotatioanl
# plt.plot(np.logspace(0.1,1.1, 5), np.logspace(2.1,3.1, 5), 'b--')   # k=1
# plt.annotate(r'$k = 1$', xy=(2, 1), xytext=(1.5,870), xycoords='data')
# plt.plot(np.logspace(-1.6,-0.6, 5), np.logspace(-0.8,1.2, 5), 'b--')  # k=1
# plt.annotate(r'$k = 2$', xy=(2, 1), xytext=(0.02,2), xycoords='data')


plt.show()

print(label)