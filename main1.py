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
    r_bl, b_bl = cv2.threshold(img_l, 150, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
    b_bl = cv2.medianBlur(b_bl, 3) #cv2.fastNlMeansDenoising(b_bl, 10, 10, 7, 21)  #  TODO: 看用哪个

    kernel = np.ones((3, 3), np.uint8)
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
    r_bl, b_bl = cv2.threshold(img_l, 150, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
    b_bl = cv2.medianBlur(b_bl, 3) #cv2.fastNlMeansDenoising(b_bl, 10, 10, 7, 21)  #  TODO: 看用哪个

    kernel = np.ones((3, 3), np.uint8)
    img_l = cv2.morphologyEx(b_bl, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀）

    b_br = img_l

    # cut the ball
    b_br = np.clip(b_br, 0, 255)  # 归一化也行
    b_br = np.array(b_br, np.uint8)
    contours, hierarchy = cv2.findContours(b_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x, y, w1, h1=cv2.boundingRect(cnt)

    r = int(max(h1, w1)/2)

    ball = b_br[y - 2:(y + 2*r + 1), x - 2:(x + 2*r + 1)]  # 注意(y,x) # 二值球
    ball_i = img[y - 2:(y + 2*r + 1), x - 2:(x + 2*r + 1)]  # 原球

    ball_i = ball

    # 给球加mask
    points = ball_i.copy()
    ballmask = np.zeros((2*r + 1, 2*r + 1), np.uint8)
    mask_ballindex = func.create_circular_mask(2*r + 3, 2*r + 3, center=[r, r], radius=r-2) #r-2  # TODO
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

path1 = 'D:\\guan2019\\2_ball\\1_pic\\60Hz\\3.5\\' #  'G:\\guan_data\\2_ball\\1_pic\\60Hz\\'  # 读入路径
filename = traversalDir_FirstDir(path1)
file_n = [path1 + name for name in filename]
print(file_n)

path2 = 'D:\\guan2019\\2_ball\\2_data\\1\\' # 'G:\\guan_data\\2_ball\\2_data\\1\\'  # 输出路径
store_n = [path2 + name + '.h5' for name in filename]
print(store_n)

for j in range(len(file_n)):  #  !!千万别错了！！！，会覆盖， #  range(2,3):#
    # images = pims.ImageSequence(file_n[j], process_func=predo)
    # images = images[1:]  # We'll take just the first 10 frames for demo purposes.

    images_o = pims.ImageSequence(file_n[j])
    images = pims.ImageSequence(file_n[j])
    i = 0
    for image in images_o:
        p = np.array(image)
        print(p)
        images[i] = pims.ImageSequence(predo(image))
        i += 1

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
        plt.figure()
        plt.imshow(img, cmap=plt.cm.gray)
        plt.title('the biggest region')
        plt.show()

# ------cut the ball
        b_br = img
        b_br = np.clip(b_br, 0, 255)  # 归一化也行
        b_br = np.array(b_br, np.uint8)
        contours, hierarchy = cv2.findContours(b_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x, y, w1, h1 = cv2.boundingRect(cnt)
        # print(x, y, w1, h1)  # region of ball
        # b_br1 = cv2.rectangle(b_br, (x-2, y-2), (x+w1+1, y+h1+1), (255, 255, 255), 1)  # 框出球
        # plt.figure()
        # plt.imshow(b_br1, cmap=plt.cm.gray)
        # plt.title('rect')
        # plt.show()

        r = int(max(h1, w1)/2)
        ball = b_br[y - 2:(y + 2*r + 1), x - 2:(x + 2*r + 1)]  # 注意(y,x) # 二值球
        ball_i = img[y - 2:(y + 2*r + 1), x - 2:(x + 2*r + 1)]  # 原球
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
        # ax0.imshow(ball_i, cmap=plt.cm.gray)
        # ax0.set_title('origin image')
        # ax1.imshow(ball, cmap=plt.cm.gray)
        # ax1.set_title('binary ball')
        # plt.show()

# ------给球加mask
        ball_i = ball
        points = ball_i.copy()
        ballmask = np.zeros((2*r + 1, 2*r + 1), np.uint8)
        mask_ballindex = func.create_circular_mask(2*r + 3, 2*r + 3, center=[r, r], radius=r-2) #r-2
        points[~mask_ballindex] = 255   # mask array
        cv2.imshow("ball_m", points)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# ------smooth the boundary of points
        blur = ((3, 3), 1)
        erode_ = (3, 3)
        dilate_ = (3, 3)
        p = cv2.dilate(cv2.erode(cv2.GaussianBlur(points, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
        ax0.imshow(points)
        ax0.set_title('origin image')
        ax1.imshow(p + points)
        ax1.set_title('p')
        plt.show()

# ------feature the points
        f = tp.locate(p, 11, invert=True)
        # print(f)  # shows the first few rows of data
        plt.figure()  # make a new figure
        tp.annotate(f, p)  # circle the points
        points_ps = f[f.columns[0:2]].values  # (y,x), np.array of dim n*2
        radius = r
        points_ps -= radius
        points_ps[:, [0, 1]] = points_ps[:, [1, 0]]  # change to (x,y)
        zps = np.sqrt(radius**2-points_ps[:, 0]**2-points_ps[:, 1]**2)
        points_ps = np.insert(points_ps, 2, values=zps, axis=1)  # (x,y,z), np.array of dim n*3
        print(points_ps)
        f_no = np.shape(np.array(f))[0]  # no. of features
        print(f_no)

# ------nomalization
        if f_no > 3:
            points_ps = points_ps[0:3, :]
            points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :]), sp.Matrix(points_ps[2, :])],
                               orthonormal=True), dtype=np.float32) * r
        elif f_no == 3:
            points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :]), sp.Matrix(points_ps[2, :])],
                               orthonormal=True), dtype=np.float32) * r
            # points_ps_n1 = spy.orthogonalize(points_ps)*radius  # 结果一致
            print(points_ps_n)#,points_ps_n1)
            # print(points_ps_n[0,0]**2+points_ps_n[0,1]**2+points_ps_n[0,2]**2,radius**2)  # (x,y,z), np.array of dim n*3
        elif f_no == 2:
            points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :])], orthonormal=True),dtype=np.float32) * r
            # print(points_ps_n)
            creat_p = np.cross(points_ps_n[0], points_ps_n[1])  # 找第三个点
            creat_p = creat_p/np.linalg.norm(creat_p)*radius
            points_ps_n = np.row_stack((points_ps_n, creat_p))
            f_no = 3
            print('normalized points')
            print(points_ps_n)


# ------points matching
        if f_no == 3:
            # find the closest points
            dis = np.zeros((f_no, 6))
            print(points_1)
            for i in range(f_no):
                dis[i, :] = list(map(lambda x: np.linalg.norm(points_ps_n[i, :]-x), points_1))  # (f_no,6)距离假定点的矩阵
            # dis = np.array([[1,2,3,4,5,6],[6,5,4,5,4,3],[1,2,3,4,5,1]])
            # print('---------------')
            print(dis)
            index_min = np.argmin(dis, axis=1)  # fearures对应前一帧/模型中的最近的点index
            # print(index_min)
            # TODO 寻找最近点有问题
            for i in range(f_no):
                for j in range(i+1, f_no):
                    dis[j, index_min[i]] = 1000
            index_min = np.argmin(dis, axis=1)
            # print(dis)
            print(index_min)
            points_co = np.array([points_1[index_min[i]] for i in range(f_no)])  # 前一帧/模型中匹配点的提取与按序排列
            # print(points_co)

        # 求旋转矩阵 及 六个点坐标  # 旋转矩阵用小数的坐标，输出用整数的
            points_11 = np.delete(points_1, index_min, axis=0)
            points_i= np.vstack((points_co, points_1))  # 前一帧/模型中的所有点（匹配点按序排列）
            print(points_co)
            index = [x for x in range(6) if x not in index_min]
            index = np.append(index_min, index)  # 六个点初始排序index记录
            print(index)
            pi = sp.Matrix(points_co).T  # 转置
            pf = sp.Matrix(points_ps_n).T
            if pi.det() != 0:  # del()不为0
                rot = pf*(pi.inv())  # 旋转矩阵
                # print('---------------')
#delete -------------
                # -# print(rot, rot.det())
                # -# points_f = np.vstack(np.hsplit(np.array(rot*sp.Matrix(points_1).T), 2))  # 旋转后的六个点坐标
                # -# points_f = np.array([points_f[i] for i in index], dtype=np.float)  # 按初始顺序排列六个点坐标
                # -# print(points_f)
                # -# print('{{{{{')
                points_ps_n1 = np.zeros(shape=(6,3))
                for i in range(len(index_min)):
                    points_ps_n1[index_min[i]] = np.array(points_ps_n[i], dtype=np.float)  # 按初始顺序排列六个点坐标
                    points_ps_n1[5-index_min[i]] = -np.array(points_ps_n[i], dtype=np.float)
                points_f = points_ps_n1  # np.vstack((points_ps_n,-points_ps_n))
                print(points_f)
        # #   可用points_co——points_ps_n，vec——vec1验证rot
        #         vec1 = [points_ps_n[0] - points_ps_n[1], points_ps_n[1] - points_ps_n[2], points_ps_n[2] - points_ps_n[0]]
        #         vec = [points_co[0] - points_co[1], points_co[1] - points_co[2], points_co[2] - points_co[0]]
        #         print(rot*sp.Matrix(vec).T,sp.Matrix(vec1).T)

            names = ['center', 'points', 'matrix']
            info = tuple(list([[x+r, y+r], points_f.flatten(), np.array(np.asarray(rot), dtype=float).reshape(-1)])) # points和matrix都变成一维的了，顺序按行读
            # print(info)
            info_dict = dict(zip(names, info))
            # print(info_dict)

        c += 1
        # print(info_dict)

        store.append(key='center', value=pd.DataFrame((info_dict['center'],)))  # 参数输出
        store.append(key='points', value=pd.DataFrame((info_dict['points'],)))
        store.append(key='matrix', value=pd.DataFrame((info_dict['matrix'],)))
        print(c)
        points_1 = points_f

    store.close()
    print('--')
    print(j)
    del store, img