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
path2 = 'D:\\guan2019\\2_ball\\2_data_new\\60Hz_rot_selected\\' # 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'  # TODO: 注意 这里只用_copy文件夹的数据！！！！

filename = [name for name in os.listdir(path2)]
pdf_rot_dict = {}
ys = [i +  (i ) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
step = 1  #30

for j in range(len(filename)):  #3,4):    # len(filename)-1,
    path3 = path2 + filename[j] + '\\'
    filename1 = [os.path.splitext(name)[0] for name in os.listdir(path3)]
    file_n = [path3 + name + '.h5' for name in filename1]
    print(filename1, file_n)

    d_theta = []
    d_rot_energy = []
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
        print(type(frame_name))

    # # translational-----------------------------------------------------------------------------------------

        # # traj pic
        # traj = pd.DataFrame({'t': np.linspace(0, max_time, N), 'x': center[:, 0], 'y': center[:, 1]})
        # print(traj.head())
        # ax = traj.plot(x='x', y='y', alpha=0.6, legend=False, title='trajectory')
        # ax.set_xlim(0, 480)  # (traj['x'].min()-10, traj['x'].max()+10)
        # ax.set_ylim(0, 480)  # (traj['y'].min()-10, traj['y'].max()+10)
        # ax.set_xlabel('x(pixel)')
        # ax.set_ylabel('y(pixel)')
        # ax.plot()
        # plt.show()
        # fig = ax.get_figure()
        # # fig.savefig(frame_traj[i])
        # del ax, fig




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
        deltatheta = points_theta
        deltaeuler = deltaeuler / math.pi * 180.0  # euler角度

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
        deltaeuler = all[4:7,:].T # euler (degree)
        quaternions = all[7:11,:].T  # quaternions
        rot_energy = 0.1*0.5*9.028*deltatheta**2 # uJ
        deltatheta = deltatheta/math.pi * 180.0  # 每次旋转的角度 (degree)

        # # -------- 单个点位移 --计算
        deltar = np.diff(points_reshape[:, 0, :], axis=0)  #------------单个点位移
        deltapr = np.sqrt(np.sum(deltar ** 2, axis=1))  # 每次旋转的球面距离(pixel)
        index = np.where(deltapr == 0)
        deltapr = np.delete(deltapr, index[0])

        d_theta = np.hstack((d_theta, deltatheta)) #全正theta
        d_rot_energy = np.hstack((d_rot_energy, rot_energy)) # rot energy
        d_axis = np.vstack((d_axis, deltaaxis))[1:,:]  # 去掉一开始那个[0,0,0]
        d_euler = np.vstack((d_euler, deltaeuler))[1:,:]  # 去掉一开始那个[0,0,0]
        d_pr = np.hstack((d_pr, deltapr))

        index = np.where(d_axis[:,2] <= 0)  # 正负theta（轴z+）
        d_new_theta = d_theta.copy()
        d_new_theta[index] = d_new_theta[index]*(-1)  # 把瞬时轴指向z=0以下的旋转角设为负
        d_new_axis = d_axis.copy()  # d_axis旋转轴指向z>0/<0都有，d_theta只有顺时针
        d_new_axis[index] = d_new_axis[index]*(-1)  # d_new_axis旋转轴全部指向z>0，d_new_theta有顺/逆时针

        d_quaternions = np.vstack((d_quaternions,quaternions)) # 四元数



        # # --------------------------------------------------------------------------------------------------------------------




        # # ---- calculate -----------------------------------------------------------------------------------------------------

        # # -- 作图rot_traj pic
        # points_traj = pd.DataFrame(
        #     {'t': np.linspace(0, max_time, N), 'x': points_reshape[:, 0,0], 'y': points_reshape[:, 0,1], 'z': points_reshape[:, 0,2]})
        # # 更改timestep-----------------------
        # px = points_traj['x'][::50]
        # py = points_traj['y'][::50]
        # pz = points_traj['z'][::50]
        # # Make ball
        # br = np.floor(np.sqrt(px[0] ** 2 + py[0] ** 2 + pz[0] ** 2)) - 2
        # bu = np.linspace(0, 2 * np.pi, 1000)
        # bv = np.linspace(0, 2 * np.pi, 1000)
        # bx = br * np.outer(np.cos(bu), np.sin(bv))
        # by = br * np.outer(np.sin(bu), np.sin(bv))
        # bz = br * np.outer(np.ones(np.size(bu)), np.cos(bv))
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(bx, by, bz, color='w', alpha=0.1)
        # for k in range(0,len(pz) - 1): # range(0,len(pz) - 1):
        #     im = ax.plot(px[k:k + 2], py[k:k + 2], pz[k:k + 2], color=plt.cm.jet(int(k * 255 / (len(pz)-1))),alpha=0.5)
        # plt.show()
        # fig = ax.get_figure()
        # del ax, fig

        # -----------
        # 轴指向z+ 的角度的fluctuation
        #  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
        # fig = plt.figure()
        # ax1 = plt.subplot(111)
        # time = np.around(np.arange(0, (len(d_new_theta) ) * 1 / 150, 1 / 150), decimals=3) #np.arange(0, (len(d_new_theta) ) ))#
        # plt.plot(time,d_new_theta, '.',markersize=1, alpha=0.4, color=colors[i], label=filename[i] + 'g')  # /180.0*math.pi
        # # ax1.set_yscale('log')
        # # ax1.set_xscale('log')
        # ax1.set_title(r' Angular Displacement' + ' [60Hz] ', fontsize=10)  # the Value of
        # ax1.set_ylabel(r'$\Delta \Theta(°)$')
        # ax1.set_xlabel(r'$time~(s)$')
        # ax1.plot()
        # plt.show()
        # del fig, ax1

        # # # -- msd
        # # dt = max_time/N
        # # msd = compute_msd(traj, t_step=dt, coords=['x', 'y'])
        # # print(msd.head())
        # # ax = msd.plot(x="tau", y="msds", logx=True, logy=True, legend=False, title='MSD')
        # # ax.fill_between(msd['tau'], msd['msds'] - msd['msds_std'], msd['msds'] + msd['msds_std'], alpha=0.2)
        # # ax.plot()
        # # plt.show()
        # # msd_i = pd.HDFStore(frame_msd[i], complib='blosc')
        # # msd_i.append(frame_name, msd, format='t', data_columns=True)
        # # fig = ax.get_figure()
        # # # fig.savefig(frame_msd_pic[i])
        # # del msd, msd_i, ax, fig



  # pr
    weights_pr = np.ones_like(d_pr) / float(len(d_pr))
    cuts_m_pr = int(max(-min(d_pr), max(d_pr)) / 0.5 // 2) + 0.5
    cuts_pr = np.arange(0, cuts_m_pr, 1)
    p_pr, x_pr, cu = plt.hist(d_pr, cuts_pr, histtype='bar', facecolor='yellowgreen', weights=weights_pr, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
    pdf_rot_dict[filename[j]+'p_pr'] = p_pr
    x_pr = (x_pr[:-1] + x_pr[1:]) / 2.
    pdf_rot_dict[filename[j]+'x_pr'] = x_pr  # 存入dict

# axis-angle-----------------------------------------------------------------------
  # 全正theta --用了两种bar来画
    weights_theta = np.ones_like(d_theta) / float(len(d_theta))
    cuts_m_theta = int(max(-min(d_theta), max(d_theta)) / 0.5 // 2) + 0.5
    cuts_theta = np.arange(-0.5, 30, 1) # 轴指向z-z+都有的theta # 另一种bin  np.arange(0, 30, 1)
    p_theta, x_theta, cu = plt.hist(d_theta, cuts_theta, histtype='bar', facecolor='yellowgreen', weights=weights_theta, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
    pdf_rot_dict[filename[j]+'p_theta'] = p_theta
    x_theta = (x_theta[:-1] + x_theta[1:]) / 2.
    pdf_rot_dict[filename[j]+'x_theta'] = x_theta  # 存入dict

  # 正负theta（轴z+）--用了两种bar来画
    weights_new_theta = np.ones_like(d_new_theta) / float(len(d_new_theta))
    cuts_m_new_theta = int(max(-min(d_new_theta), max(d_new_theta)) / 0.5 // 2) + 0.5
    cuts_new_theta = np.arange(-30.25, 30,0.5) # np.arange(-29, 30, 2)  # np.arange(-30.25, 30, 0.5)  # 轴指向z+的theta # 另一种bin
    p_new_theta, x_new_theta, cu = plt.hist(d_new_theta,cuts_new_theta, histtype='bar', facecolor='yellowgreen', weights=weights_new_theta, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
    pdf_rot_dict[filename[j]+'p_new_theta'] = p_new_theta
    x_new_theta = (x_new_theta[:-1] + x_new_theta[1:]) / 2.
    pdf_rot_dict[filename[j]+'x_new_theta'] = x_new_theta  # 存入dict

# rot energy
    weights_rot_energy = np.ones_like(d_rot_energy) / float(len(d_rot_energy))
    print(max(d_rot_energy),min(d_rot_energy))
    cuts_m_rot_energy = int(max(-min(d_rot_energy), max(d_rot_energy)) / 0.5 // 2) + 0.5
    cuts_rot_energy = np.arange(0,0.5,0.001)  # 轴指向z+的theta # 另一种bin np.arange(-30, 30, 1)
    p_rot_energy, x_rot_energy, cu = plt.hist(d_rot_energy,cuts_rot_energy, histtype='bar', facecolor='yellowgreen',weights=weights_rot_energy, alpha=0.75, rwidth=1)#, density=True)  # au是counts，bu是deltar
    pdf_rot_dict[filename[j] + 'p_rot_energy'] = p_rot_energy
    x_rot_energy = (x_rot_energy[:-1] + x_rot_energy[1:]) / 2.
    pdf_rot_dict[filename[j] + 'x_rot_energy'] = x_rot_energy  # 存入dict

# # euler----------------------------------------------------------------------------
#   # euler1
#     d_euler1 = d_euler[:,0]
#     weights_euler1 = np.ones_like(d_euler1) / float(len(d_euler1))
#     cuts_m_euler1 = int(max(-min(d_euler1), max(d_euler1)) / 0.5 // 2)# + 0.5
#     cuts_euler1 = np.arange(-30.5, 30, 1)
#     p_euler1, x_euler1, cu = plt.hist(d_euler1, cuts_euler1, histtype='bar', facecolor='yellowgreen', weights=weights_euler1, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
#     pdf_rot_dict[filename[j]+'p_euler1'] = p_euler1
#     x_euler1 = (x_euler1[:-1] + x_euler1[1:]) / 2.
#     pdf_rot_dict[filename[j]+'x_euler1'] = x_euler1  # 存入dict
#   # euler2
#     d_euler2 = d_euler[:,1]
#     weights_euler2 = np.ones_like(d_euler2) / float(len(d_euler2))
#     cuts_m_euler2 = int(max(-min(d_euler2), max(d_euler2)) / 0.5 // 2)# + 0.5
#     cuts_euler2 = np.arange(-30.5, 30, 1)
#     p_euler2, x_euler2, cu = plt.hist(d_euler2, cuts_euler2, histtype='bar', facecolor='yellowgreen', weights=weights_euler2, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
#     pdf_rot_dict[filename[j]+'p_euler2'] = p_euler2
#     x_euler2 = (x_euler2[:-1] + x_euler2[1:]) / 2.
#     pdf_rot_dict[filename[j]+'x_euler2'] = x_euler2  # 存入dict
#   # euler3
#     d_euler3 = d_euler[:,2]
#     weights_euler3 = np.ones_like(d_euler3) / float(len(d_euler3))
#     cuts_m_euler3 = int(max(-min(d_euler3), max(d_euler3)) / 0.5 // 2)# + 0.5
#     cuts_euler3 = np.arange(-30.5, 30, 1)
#     p_euler3, x_euler3, cu = plt.hist(d_euler3, cuts_euler3, histtype='bar', facecolor='yellowgreen', weights=weights_euler3, alpha=0.75, rwidth=1, density=True)  # au是counts，bu是deltar
#     pdf_rot_dict[filename[j]+'p_euler3'] = p_euler3
#     x_euler3 = (x_euler3[:-1] + x_euler3[1:]) / 2.
#     pdf_rot_dict[filename[j]+'x_euler3'] = x_euler3  # 存入dict
#
# # quaternions-----------------------------------------------------------
#     bing1 = np.arange(0,1.004,0.005)
#     bing2 = np.arange(-0.09-0.0025,0.093,0.005)
#     # quaternion1
#     d_quat1 = d_quaternions[:,0]
#     au, bu, cu = plt.hist(d_quat1, bing1, histtype='bar', facecolor='yellowgreen', alpha=0.75,rwidth=0.01)  # , density=True)  # au是counts，bu是deltar
#     au /= len(d_quat1)
#     bu = (bu[:-1] + bu[1:]) / 2.
#     pdf_rot_dict[filename[j] + 'x_quat1'] = bu  # 存入dict
#     pdf_rot_dict[filename[j] + 'p_quat1'] = au  # - np.min(au1)
#     # quaternion2
#     d_quat2 = d_quaternions[:,1]
#     au, bu, cu = plt.hist(d_quat2, bing2, histtype='bar', facecolor='yellowgreen', alpha=0.75,rwidth=0.01)  # , density=True)  # au是counts，bu是deltar
#     au /= len(d_quat1)
#     bu = (bu[:-1] + bu[1:]) / 2.
#     pdf_rot_dict[filename[j] + 'x_quat2'] = bu  # 存入dict
#     pdf_rot_dict[filename[j] + 'p_quat2'] = au  # - np.min(au1)
#     # quaternion3
#     d_quat3 = d_quaternions[:,2]
#     au, bu, cu = plt.hist(d_quat3, bing2, histtype='bar', facecolor='yellowgreen', alpha=0.75,rwidth=0.01)  # , density=True)  # au是counts，bu是deltar
#     au /= len(d_quat1)
#     bu = (bu[:-1] + bu[1:]) / 2.
#     pdf_rot_dict[filename[j] + 'x_quat3'] = bu  # 存入dict
#     pdf_rot_dict[filename[j] + 'p_quat3'] = au  # - np.min(au1)
#     # quaternion4
#     d_quat4 = d_quaternions[:,3]
#     au, bu, cu = plt.hist(d_quat4, bing2, histtype='bar', facecolor='yellowgreen', alpha=0.75,rwidth=0.01)  # , density=True)  # au是counts，bu是deltar
#     au /= len(d_quat1)
#     bu = (bu[:-1] + bu[1:]) / 2.
#     pdf_rot_dict[filename[j] + 'x_quat4'] = bu  # 存入dict
#     pdf_rot_dict[filename[j] + 'p_quat4'] = au  # - np.min(au1)

# -----------pdf汇总作图-------------


# # -- rot_pdf
# # 点的r的pdf
# # path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_copy\\'
# fig = plt.figure()
# ax0 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_pr'] / 480.0 * 260, pdf_rot_dict[filename[i] + 'p_pr'], alpha=0.5,
#     #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_pr'] / 480.0 * 260, pdf_rot_dict[filename[i] + 'p_pr'], 'o-',alpha=0.5,
#                 color=colors[i], label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# plt.legend(label)
# ax0.set_title('PDF -- the Value of Single Point\'s Displacement' + ' [60Hz] ', fontsize=10)
# ax0.set_xlabel('$\Delta R(mm)$')  # ax0.set_xlabel('deltaR(pixel)')
# ax0.set_ylabel('P')
# ax0.plot()
# plt.show()
# del ax0, fig

# -----------
# 轴指向z+z- 的角度的pdf
#  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
# fig = plt.figure()
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,
#     #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], 'o-',alpha=0.5,
#                 color=colors[i],  label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# plt.legend(label)
# ax1.set_title('PDF -- the Value of Angular Displacement' + ' [60Hz] ', fontsize=10)  # the Value of
# ax1.set_xlabel('$\Delta \Theta(°)$')
# ax1.set_ylabel('P')
# ax1.plot()
# plt.show()
# del fig, ax1

#-----------
# 轴指向z+ 的角度的pdf
#  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
fig = plt.figure()
ax1 = plt.subplot(111)
label = []
for i in range(len(filename)):  # 2,3):    #
    # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,
    #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
    plt.plot(pdf_rot_dict[filename[i] + 'x_new_theta'], pdf_rot_dict[filename[i] + 'p_new_theta'], 'o-',alpha=0.4,
                color=colors[i],  label=filename[i] + 'g')  # /180.0*math.pi
    label.append(filename[i] + 'g')
ax1.legend(label)
leg = ax1.legend(label)
leg.get_frame().set_linewidth(0.0)
# ax1.set_yscale('log')
# ax1.set_xscale('log')
ax1.set_title(r'PDF -- Angular Displacement' + ' [60Hz] ', fontsize=10)  # the Value of
ax1.set_xlabel(r'$\Delta \Theta(°)$')
ax1.set_ylabel(r'$P(\Delta \Theta)$')
ax1.plot()
plt.show()
del fig, ax1



# # rotation energy 的pdf
# #  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
# fig = plt.figure()
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,
#     #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_rot_energy'], pdf_rot_dict[filename[i] + 'p_rot_energy'], 'o',alpha=0.5,
#                 color=colors[i],  label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# ax1.legend(label)
# leg = ax1.legend(label)
# leg.get_frame().set_linewidth(0.0)
# ax1.set_title(r'PDF -- rotational energy' + ' [60Hz] ', fontsize=10)  # the Value of
# ax1.set_xlabel(r'$E_R~(\mu J)$')
# ax1.set_ylabel(r'$P(E_R)$')
# ax1.set_yscale('log')
# ax1.plot()
# plt.show()
# del fig, ax1



# #-----------
# # euler1的pdf
# #  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
# fig = plt.figure()
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,
#     #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_euler1'], pdf_rot_dict[filename[i] + 'p_euler1'], 'o-',alpha=0.5,
#                 color=colors[i],  label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# plt.legend(label)
# ax1.set_title(r'PDF -- Euler $\alpha$' + ' [60Hz] ', fontsize=10)  # the Value of
# ax1.set_xlabel(r'$\Delta \alpha(°)$')
# ax1.set_ylabel('P')
# ax1.plot()
# plt.show()
# del fig, ax1
#
# # euler2的pdf
# #  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
# fig = plt.figure()
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,
#     #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_euler2'], pdf_rot_dict[filename[i] + 'p_euler2'], 'o-',alpha=0.5,
#                 color=colors[i],  label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# plt.legend(label)
# ax1.set_title(r'PDF -- Euler $\beta $' + ' [60Hz] ', fontsize=10)  # the Value of
# ax1.set_xlabel(r'$\Delta \beta (°)$')
# ax1.set_ylabel('P')
# ax1.plot()
# plt.show()
# del fig, ax1
#
# # euler3的pdf
# #  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
# fig = plt.figure()
# ax1 = plt.subplot(111)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,
#     #             color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_euler3'], pdf_rot_dict[filename[i] + 'p_euler3'], 'o-',alpha=0.5,
#                 color=colors[i],  label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# plt.legend(label)
# ax1.set_title(r'PDF -- Euler $\gamma$' + ' [60Hz] ', fontsize=10)  # the Value of
# ax1.set_xlabel(r'$\Delta \gamma(°)$')
# ax1.set_ylabel('P')
# ax1.plot()
# plt.show()
# del fig, ax1

#
# # #-----------
# # quaternion的pdf
# #  path2 = 'D:\\guan2019\\2_ball\\2_data\\60Hz_select_rot\\'
# fig = plt.figure()
# ax1 = plt.subplot(221)
# label = []
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_quat1'], pdf_rot_dict[filename[i] + 'p_quat1'], 'o-',alpha=0.5, color=colors[i],  label=filename[i] + 'g')
#     label.append(filename[i] + 'g')
# plt.legend(label)
# # ax1.set_title(r'PDF -- quaternion1' + ' [60Hz] ', fontsize=10)  # the Value of
# # ax1.set_xlabel(r'$q_1$')
# ax1.set_ylabel(r'$P_{q_1}$')
# ax1.set_yscale('log')
# ax1.set_ylim((0.0001,1))
# ax1.set_xlim((0.86,1.015))
# ax1.legend(label)
# leg = ax1.legend(label)
# leg.get_frame().set_linewidth(0.0)
# ax1.plot()
# ax2 = plt.subplot(222)
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_quat2'], pdf_rot_dict[filename[i] + 'p_quat2'], 'o-',alpha=0.5,color=colors[i],  label=filename[i] + 'g')
# # plt.legend(label)
# # ax1.set_title(r'PDF -- quaternion2' + ' [60Hz] ', fontsize=10)  # the Value of
# # ax2.set_xlabel(r'$q_2$')
# ax2.set_ylabel(r'$P_{q_2}$')
# ax2.set_yscale('log')
# ax2.set_ylim((0.0001,1))
# ax2.plot()
# ax3 = plt.subplot(223)
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_quat3'], pdf_rot_dict[filename[i] + 'p_quat3'], 'o-',alpha=0.5,color=colors[i],  label=filename[i] + 'g')
# # plt.legend(label)
# # ax3.set_title(r'PDF -- quaternion3' + ' [60Hz] ', fontsize=10)  # the Value of
# # ax3.set_xlabel(r'$q_3$')
# ax3.set_ylabel(r'$P_{q_3}$')
# ax3.set_yscale('log')
# ax3.set_ylim((0.0001,1))
# ax3.plot()
# ax4 = plt.subplot(224)
# for i in range(len(filename)):  # 2,3):    #
#     # plt.scatter(pdf_rot_dict[filename[i] + 'x_theta'], pdf_rot_dict[filename[i] + 'p_theta'], alpha=0.5,color=colors[i], cmap='hsv', label=filename[i] + 'g')
#     plt.plot(pdf_rot_dict[filename[i] + 'x_quat4'], pdf_rot_dict[filename[i] + 'p_quat4'], 'o-',alpha=0.5,color=colors[i], label=filename[i] + 'g')
# # plt.legend(label)
# # ax4.set_title(r'PDF -- quaternion4' + ' [60Hz] ', fontsize=10)  # the Value of
# # ax4.set_xlabel(r'$q_4$')
# ax4.set_ylabel(r'$P_{q_4}$')
# ax4.set_yscale('log')
# ax4.set_ylim((0.0001,1))
# ax4.plot()
#
# plt.show()
# # del fig, ax1