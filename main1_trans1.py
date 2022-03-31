# 主程序
# 从maintry.py获得 第一帧坐标 & 修改其他参数

# TODO 获得正交的三个点坐标，计算旋转矩阵，得到的六个点坐标不是正交的；1个点2个点目前都是复制的

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, feature, transform, draw
import skimage
import trackpy as tp
import pims
import warnings
import sympy as sp
import scipy.signal as signal
import spectral as spy
import os.path
import glob
import pandas as pd
import func
from matplotlib.patches import Ellipse
from ellipse import LsqEllipse
import functools
import matplotlib.cm as cm
from scipy import optimize

def traversalDir_FirstDir(path):  # 返回一级子文件夹名字
    list = []
    if (os.path.exists(path)):    #获取该目录下的所有文件或文件夹目录路径
        files = glob.glob(path + '\\*' )
        # print(files)
        for file in files:            #判断该路径下是否是文件夹
            if (os.path.isdir(file)):                #分成路径和文件的二元元组
                h = os.path.split(file)
                # print(h[1] )
                list.append(h[1])
        return list

@pims.pipeline
def predo(img):
    if len(img.shape) == 3:
        img = img[:, :, 0]
    h, w = img.shape[:2]
    r_m = 4.9 / 10. * h
    mask_index = func.create_circular_mask(h, w, center=[h / 2+2, w / 2-2],
                                                radius=r_m - 12)  # TODO: change the region of the stage#func.create_circular_mask(h, w, center=[h / 2 + 6.5, w / 2 - 2], radius=r_m - 1)  # TODO: change the region of the stage
    img_m = img.copy()
    img_m[~mask_index] = img.mean()  # image array with mask
    mask = np.zeros((h, w), np.uint8)
    mask[mask_index] = 0  # mask array

    # stretch the masked image
    img_g = cv2.GaussianBlur(img_m, (3, 3), 0)
    img_l = func.grey_scale(img_m)  # func_ball.log(43, img_g)  # log拉伸
    img_l[~mask_index] = img_l.mean()  # image array with mask

    # 二进制阈值化处理
    r_bl, b_bl = cv2.threshold(img_l, 168, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
    b_bl = cv2.medianBlur(b_bl, 3) #cv2.fastNlMeansDenoising(b_bl, 10, 10, 7, 21)  #  TODO: 看用哪个

    kernel = np.ones((4,4), np.uint8)  # TODO: change
    img_l = cv2.morphologyEx(b_bl, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀）
    return img_l

# 本文件为预运行函数，用于获得初始点坐标
def points_ini(img):
    # add a mask for the container
    h, w = img.shape[:2]
    r_m = 4.9 / 10. * h
    mask_index = func.create_circular_mask(h, w, center=[h / 2 +2, w / 2 -2], radius=r_m - 12)  # TODO: change the region of the stage#func.create_circular_mask(h, w, center=[h / 2 + 6.5, w / 2 - 2], radius=r_m - 1)  # TODO: change the region of the stage
    img_m = img.copy()
    img_m[~mask_index] = img.mean()    # image array with mask
    mask = np.zeros((h, w), np.uint8)
    mask[mask_index] = 0  # mask array

    # stretch the masked image
    img_g = cv2.GaussianBlur(img_m, (3, 3), 0)
    img_l = func.grey_scale(img_m)  #  func_ball.log(43, img_g)  # log拉伸
    img_l[~mask_index] = img_l.mean()    # image array with mask

    # 二进制阈值化处理 # 此处粗略，点不需要全，保证球形状完整即可
    r_bl, b_bl = cv2.threshold(img_l, 160, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
    b_bl = cv2.medianBlur(b_bl, 3) #cv2.fastNlMeansDenoising(b_bl, 10, 10, 7, 21)  #  TODO: 看用哪个

    kernel = np.ones((3,3), np.uint8)  # TODO: change
    img_l = cv2.morphologyEx(b_bl, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀）

    b_br = img_l

    # cut the ball
    b_br = np.clip(b_br, 0, 255)  # 归一化也行
    b_br = np.array(b_br, np.uint8)
    contours, hierarchy = cv2.findContours(b_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w1, h1=cv2.boundingRect(cnt)

    r = int(max(h1, w1)/2)

    ball = b_br[y - 1:(y + 2*r + 1), x - 3:(x + 2*r - 1)]  # 注意(y,x) # 二值球 TODO 根据框的位置改
    ball_i = img[y - 1:(y + 2*r + 1), x - 3:(x + 2*r - 1)]  # 原球 TODO 根据框的位置改

    ball_i = ball

    # 给球加mask
    points = ball_i.copy()
    ballmask = np.zeros((2*r + 1, 2*r + 1), np.uint8)
    mask_ballindex = func.create_circular_mask(2*r + 2, 2*r + 2, center=[r+1, r+1], radius=r-2) #r-2  # TODO
    points[~mask_ballindex] = 255  # ball_m[mask_ballindex].mean()  # mask array

    # smooth the boundary of points
    blur = ((3, 3), 1)
    erode_ = (3, 3)
    dilate_ = (3, 3)
    p = cv2.dilate(cv2.erode(cv2.GaussianBlur(points, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))

    # feature the points
    f = tp.locate(p, 11, invert=True)  # TODO

    points_ps = f[f.columns[0:2]].values  # (y,x), np.array of dim n*2
    radius = r
    points_ps -= radius
    points_ps[:,[0, 1]] = points_ps[:,[1, 0]]  # change to (x,y)
    zps = np.sqrt(radius**2-points_ps[:, 0]**2-points_ps[:, 1]**2)
    points_ps = np.insert(points_ps, 2, values=zps, axis=1)  # (x,y,z), np.array of dim n*3
    f_no = np.shape(np.array(f))[0]  # no. of features
    if f_no >= 3:
        points_ps = points_ps[0:3, :]
        f_no = 3

    # normalizing
    if f_no == 3:
        points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points_ps[0,:]),sp.Matrix(points_ps[1,:]),sp.Matrix(points_ps[2,:])], orthonormal=True), dtype=np.float32)*r
        # points_ps_n1 = spy.orthogonalize(points_ps)*radius  # 结果一致
    # -----------------------------------------------------------------------------
        return np.vstack((points_ps_n,-1*points_ps_n))  # 输出正交化的坐标
    #------------------------------------------------------------------------------
    if f_no < 3:
        return np.vstack((points_ps, -1 * points_ps))  # 输出正交化的坐标
        print('points are less than 3')


# "TODO" labels parameters to be changed.
# "todo" labels code not finished yet

warnings.filterwarnings('ignore')

# path1 = 'D:\\guan2019\\2_ball\\1_pic\\5g\\'  # 读入路径
path1 = 'D:\\guan2019\\2_ball\\1_pic\\60Hz\\6.0\\'  # 读入路径
filename = traversalDir_FirstDir(path1)
file_n = [path1 + name for name in filename]
print(file_n)

# path2 = 'D:\\guan2019\\2_ball\\2_data\\5g_t\\'  # 输出路径
path2 = 'D:\\guan2019\\2_ball\\2_data\\60tt\\6.0\\'  # 输出路径
store_n = [path2 + name + '.h5' for name in filename]
print(store_n)

for j in range(len(file_n)): #  !!千万别错了！！！，会覆盖， #  range(2,3):#
    images = pims.ImageSequence(file_n[j])#process_func=predo)
    images = predo(images)
    # images = images[1:]  # We'll take just the first 10 frames for demo purposes.

    store =pd.HDFStore(store_n[j], complib='blosc')
    c = 0
    points_1 = points_ini(images[0])
    if points_1.shape[0] < 6:
        for num in range(1,len(images)):
            points_1 = points_ini(images[num])
            if points_1.shape[0] == 6:
                print(num)
                break

    print(points_1)

    for img in images[1:]:
        img = np.array(img)
        h, w = img.shape[:2]
        # plt.figure()
        # plt.imshow(img, cmap=plt.cm.gray)
        # plt.title('the biggest region')
        # plt.show()

# ------cut the ball
        b_br = img
        b_br = np.clip(b_br, 0, 255)  # 归一化也行
        # b_br = np.array(b_br, np.uint8)
        contours, hierarchy = cv2.findContours(b_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0][:,0,:]
        (x, y), radius = cv2.minEnclosingCircle(cnt) # 最小外接圆 （x,y）圆心， radius半径


        def plot_all(residu2=False):
            plt.figure(facecolor='white')  # figsize=(7, 5.4), dpi=72,
            plt.axis('equal')

            theta_fit = np.linspace(-np.pi, np.pi, 180)

            x_fit2 = x + radius* np.cos(theta_fit)
            y_fit2 = y+ radius* np.sin(theta_fit)
            plt.plot(x_fit2, y_fit2, 'k--', lw=2)
            plt.plot([x], [y], 'gD', mec='r', mew=1)

            # 数据
            plt.plot(cnt[:,0], cnt[:,1], 'ro', label='data', ms=8, mec='b', mew=1)
            plt.legend(loc='best', labelspacing=0.1)

        # plot_all(residu2=True)
        # plt.show()



        c += 1
        names = ['center', ]
        info = [[x, y], ]
        info_dict = dict(zip(names, info))

        store.append(key='center', value=pd.DataFrame((info_dict['center'],)))  # 参数输出
        # print(c)


    store.close()
    print('--')
    print(j)
    del store