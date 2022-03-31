# 主程序，获得球心位置，点位置，旋转矩阵
# 给363上/code_ball/main2_1122.py debug用，服务器获得的数据下载在\guan2019\2_ball\2_data_new\，两个文件除了路径完全一样

# TODO 获得正交的三个点坐标，计算旋转矩阵，得到的六个点坐标不是正交的；1个点是复制的；记录点在球面的位置都用1来归一化半径

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
import math
import func_ball0913

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


def ButterworthPassFilter(image,d,n):
    """
    butterworth高通滤波器
    """
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    s1 = np.log(np.abs(fshift))
    def make_transform_matrix(d):
        transform_matrix = np.zeros(image.shape)
        center_point = tuple(map(lambda x:(x-1)/2,s1.shape))
        for i in range(transform_matrix.shape[0]):
            for j in range(transform_matrix.shape[1]):
                def cal_distance(pa,pb):
                    from math import sqrt
                    dis = sqrt((pa[0]-pb[0])**2+(pa[1]-pb[1])**2)
                    return dis
                dis = cal_distance(center_point,(i,j))
                transform_matrix[i,j]=1/(1+(dis/d)**(2*n))
        return transform_matrix
    d_matrix = make_transform_matrix(d)
    d_matrix = d_matrix+0.5
    new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift*d_matrix)))
    return new_img


# 本文件为预运行函数，用于获得初始点坐标
def points_ini(img):
    # 给container加mask，圈球，标圆心
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度影象
    # gray = 1.5*gray # TODO 增加对比度
    # gray = ButterworthPassFilter(gray,10,3)
    # gray = 1.5*gray # TODO 增加对比度
    # gray[gray>255]=255
    # gray = gray.astype(np.uint8) 
    circle0 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=10, minRadius=500, maxRadius=600)  # TODO 检测容器
    if len(circle0[0])==1:
        x,y,r = circle0[0][0][0],circle0[0][0][1],circle0[0][0][2]
    # print(math.floor(x-r),math.ceil(x+r), math.floor(y-r),math.ceil(y+r))
    gray = gray[ math.floor(y-r):math.ceil(y+r),math.floor(x-r):math.ceil(x+r)]
    img = gray.copy() 


    h,w = gray.shape[0], gray.shape[1] # container的尺寸 x对应height，y对应width
    mask_contianer = func_ball0913.create_circular_mask(h,w, center=[w/2, h/2], radius=min(h,w)/2-3)  # TODO 加r-0.5的mask，遮挡球边阴影
    gray = 1.4*gray
    # gray = ButterworthPassFilter(gray,10,3)  # todo
    gray[~mask_contianer] = 255
    gray[gray>255]=255
    gray = gray.astype(np.uint8) 

    circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,1000, param1=50, param2=10, minRadius=54, maxRadius=65)  # TODO 把半徑範圍縮小點，檢測內圓，瞳孔
    if len(circle1[0])==1:
        x,y,r = circle1[0][0][0],circle1[0][0][1],circle1[0][0][2]
    # aa=2 # hough变换总是偏一点，增加矫正参数
    # x,y,r = int(x),int(y-aa),int(r+aa/2)


    # 提取球表面点
    ball_o = img.copy()
    ball_o = ball_o[math.floor(y-r):math.ceil(y+r),math.floor(x-r):math.ceil(x+r)] #切下球部分，Frame类型,
    ball_amp = 1.4*ball_o # TODO 增加对比度
    # ball_amp = ButterworthPassFilter(ball_amp,8,2) # todo
    h,w = math.ceil(x+r)-math.floor(x-r), math.ceil(y+r)-math.floor(y-r) # ball_cut的尺寸 x对应height，y对应width
    mask_ballindex = func_ball0913.create_circular_mask(ball_o.shape[0],ball_o.shape[1], center=[r, r], radius=r-1.5)  # TODO 加r-0.5的mask，遮挡球边阴影
    ball_amp[~mask_ballindex] = 0
    ball_amp[ball_amp>255]=255
    # ball_amp[ball_amp<100]=0 #TODO
    ball_amp = np.round(ball_amp) #球黑点白(后面feature椭圆要黑底白点)
    ball_amp = ball_amp.astype(np.uint8) # ball_amp增加了对比度的ball_cut
    highlightfilter = np.zeros((h,w), dtype="uint8")  # 加一个黑光源，去除高光
    strength = -100
    for m in range(h):
        for n in range(w):
            #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
            distance = math.pow((h/2.0-n), 2) + math.pow((h/2.0-m), 2)
            if (distance < (h/2.0) **2):
                result = (int)(strength*( 1.0 - math.sqrt(distance) / (h/2.0) ))#按照距离大小计算增强的光照值
                f = max(0, ball_amp[n,m] + result)#判断边界 防止越界
                highlightfilter[n,m] = np.uint8(f)
            else:
                highlightfilter[n,m] = np.uint8(ball_amp[n,m])
    ball_amp = highlightfilter
    ball_amp[ball_amp<100]=0 #TODO
    # kernel = np.ones((4,4), np.uint8)  # TODO: change
    # ball_amp = cv2.morphologyEx(ball_amp, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀））


    # 标记表面点的中心 (cv2.fitellipse)
    gray = ball_amp
    th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cnts, hiers = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    points=[]# 球表面点在ball_cut中的(x,y)坐标，默认y轴向下为正
    print(len(cnts))
    for cnt in cnts:
        if len(cnt)<5:
            continue
        else:
            ellipse = cv2.fitEllipse(cnt) # ((x,y),(a,b),theta) 中心坐标，半长轴半短轴，倾斜角
            if np.max(ellipse[1]) < 30 and np.min(ellipse[1])>3:
                points.append(np.asarray(ellipse[0]))
                cv2.ellipse(ball_o, ellipse, (255,0, 255), 1, cv2.LINE_AA)
                cv2.circle(ball_o, (round(ellipse[0][0]),round(ellipse[0][1])), 2, (255, 0, 0),1)  # 畫椭圆圓心
            else:
                continue
    # # 画图
    # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
    # ax0.imshow(ball_o, cmap='gray')
    # ax0.set_title('origin image')
    # ax1.imshow(ball_amp, cmap='gray')
    # ax1.set_title('ball')
    # plt.show()

    points = np.array(points) 
    points[:,0] -= w/2 # 把points的y坐标改成第一象限（向上为正)，圆心位于球心 
    points[:,1] -= h/2
    points[:,1] *= -1
    z = np.sqrt(r**2-points[:, 0]**2-points[:, 1]**2) #计算points的z坐标
    points = np.insert(points, 2, values=z, axis=1)  # (x,y,z), 球心为圆心的第一象限，np.array of dim n*3
    points_ps = points[~np.isnan(points).any(axis=1),:]
    # print(points_ps)
    f_no = np.shape(np.array(points_ps))[0]  # no. of features
    # print(f_no)

    if f_no >= 3:
        points_ps = points_ps[0:3, :]
        points_ps_n = np.array(
            sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :]), sp.Matrix(points_ps[2, :])],
                            orthonormal=True), dtype=np.float32)
        f_no = 3

    # normalizing
    if f_no == 3:
        points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points_ps[0,:]),sp.Matrix(points_ps[1,:]),sp.Matrix(points_ps[2,:])], orthonormal=True), dtype=np.float32)
        # points_ps_n1 = spy.orthogonalize(points)  # 结果一致
    # -----------------------------------------------------------------------------
        # print(np.vstack((points_ps_n,-1*points_ps_n)))
        return x,y,r,np.vstack((points_ps_n,-1*points_ps_n))  # 输出正交化的坐标
    #------------------------------------------------------------------------------
    elif f_no == 2:
        points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :])], orthonormal=True),dtype=np.float32)
        creat_p = np.cross(np.array(points_ps_n[0]), np.array(points_ps_n[1]),axis=0) # 找第三个点
        creat_p = creat_p/np.linalg.norm(creat_p)
        creat_p.resize((1,3))
        points_ps_n = np.concatenate((points_ps_n, creat_p))#np.row_stack((points_ps_n, creat_p))
        return x,y,r,np.vstack((points_ps_n, -1 * points_ps_n))  # 输出正交化的坐标
    else:
        return x,y,r,np.vstack((points_ps, -1 * points_ps))  # 输出正交化的坐标
        print('points are less than 3')


# "TODO" labels parameters to be changed.
# "todo" labels code not finished yet

warnings.filterwarnings('ignore')

path1 = 'D:\\guan2021\\report_b\\new\\exp\\1_frame\\50hz_240\\' #  'G:\\guan_data\\2_ball\\1_pic\\60Hz\\'  # 读入路径
filename = traversalDir_FirstDir(path1)
file_n = [path1 + name for name in filename]
print(file_n)

path2 = 'd:\\guan2021\\report_b\\new\\exp\\2_data\\frame\\50hz_240\\' # 'G:\\guan_data\\2_ball\\2_data\\1\\'  # 输出路径
store_n = [path2 + name + '.h5' for name in filename]
print(store_n)

for j in range(2,len(file_n)):  #  !!千万别错了！！！，会覆盖， #  range(2,3):#
    # images = pims.ImageSequence(file_n[j], process_func=predo)
    # images = images[1:]  # We'll take just the first 10 frames for demo purposes.

    images_o = pims.ImageSequence(file_n[j])
    images = pims.ImageSequence(file_n[j])
    i = 0
    # for image in images_o:
    #     p = np.array(image)
    #     print(p)
    #     images[i] = pims.ImageSequence(predo(image))
    #     i += 1

    store =pd.HDFStore(store_n[j], complib='blosc')
    c = 0
    points_1 = points_ini(images[0])[3]
    circle3 =  np.array(points_ini(images[0])[0:3])
    circle3.resize((1,1,3))
    if points_1.shape[0] < 6:
        for num in range(1,len(images)):
            points_1 = points_ini(images[num])[3]
            if points_1.shape[0] == 6:
                print(num)
                break

    print(points_1)

    for img in images[1:]:
# ------圈球，标圆心
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度影象
        circle0 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=45, minRadius=500, maxRadius=600)  # TODO 检测容器
        if circle0 is None:
            circle0 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 1000, param1=100, param2=45, minRadius=500, maxRadius=600)  # TODO 检测容器
            x,y,r = circle0[0][0][0],circle0[0][0][1],circle0[0][0][2]
            circle2=circle0
        elif len(circle0[0])==1:
            x,y,r = circle0[0][0][0],circle0[0][0][1],circle0[0][0][2]
            circle2=circle0

            
        # print(math.floor(x-r),math.ceil(x+r), math.floor(y-r),math.ceil(y+r))
        gray = gray[ math.floor(y-r):math.ceil(y+r),math.floor(x-r):math.ceil(x+r)]
        img = gray.copy()

        # plt.subplot(111), plt.imshow(gray, cmap='gray')
        # plt.title('circle on the ball'), plt.xticks([]), plt.yticks([])
        # plt.show()

        h,w = gray.shape[0], gray.shape[1] # container的尺寸 x对应height，y对应width
        mask_contianer = func_ball0913.create_circular_mask(h,w, center=[w/2, h/2], radius=min(h,w)/2-3)  # TODO 加r-0.5的mask，遮挡球边阴影
        gray = 1.4*gray  #TODO
        # gray = ButterworthPassFilter(gray,10,3)  # todo
        gray[~mask_contianer] = 255
        gray[gray>255]=255
        gray = gray.astype(np.uint8) 

        # plt.subplot(111), plt.imshow(gray, cmap='gray')
        # plt.title('container+mask'), plt.xticks([]), plt.yticks([])
        # plt.show()

        circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,1000, param1=100, param2=20, minRadius=int(math.floor(circle3[0][0][2]-1.5)), maxRadius=int(math.ceil(circle3[0][0][2]+1.5)))  # TODO 把半徑範圍縮小點，檢測內圓，瞳孔
        if circle1 is None:
            circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1,1000, param1=100, param2=15,minRadius=int(math.floor(circle3[0][0][2]-5)), maxRadius=int(math.ceil(circle3[0][0][2]+5)) ) # TODO 把半徑範圍縮小點，檢測內圓，瞳孔
            x,y,r = circle1[0][0][0],circle1[0][0][1],circle1[0][0][2]
            circle3 = circle1
        elif len(circle1[0])==1:
            x,y,r = circle1[0][0][0],circle1[0][0][1],circle1[0][0][2]
            circle3 = circle1
        else:
            circle1 = circle3
        # aa=2 # hough变换总是偏一点，增加矫正参数
        # x,y,r = int(x),int(y-aa),int(r+aa/2)
        print(x,y,r) # 球坐标&半径
        # # # 画图
        # circles = circle1[0, :, :]  # 提取為二維
        # circles = np.uint16(np.around(circles))  # 四捨五入，取整
        # img1 = img.copy()
        # if len(circles[:])==1:
        #     cv2.circle(img1, (round(x), round(y)), math.ceil(r), (255, 0, 0), 1)  # 畫圓
        #     cv2.circle(img1, (round(x), round(y)), 2, (255, 0, 0),1)  # 畫圓心
        # plt.subplot(111), plt.imshow(img1, cmap='gray')
        # plt.title('circle on the ball'), plt.xticks([]), plt.yticks([])
        # plt.show()
        #----------------------------------------------------------------------------------------------------------------------------

# ------提取球表面点
        ball_o = img.copy()
        ball_o = ball_o[math.floor(y-r):math.ceil(y+r),math.floor(x-r):math.ceil(x+r)] #切下球部分，Frame类型,
        ball_amp = 1.4*ball_o # TODO 增加对比度
        # ball_amp = ButterworthPassFilter(ball_amp,8,2) # todo
        h,w = math.ceil(x+r)-math.floor(x-r), math.ceil(y+r)-math.floor(y-r) # ball_cut的尺寸 x对应height，y对应width
        mask_ballindex = func_ball0913.create_circular_mask(ball_o.shape[0],ball_o.shape[1], center=[r, r], radius=r-1.5)  # TODO 加r-0.5的mask，遮挡球边阴影
        ball_amp[~mask_ballindex] = 0
        ball_amp[ball_amp>255]=255
        # ball_amp[ball_amp<100]=0 #TODO
        ball_amp = np.round(ball_amp) #球黑点白(后面feature椭圆要黑底白点)
        ball_amp = ball_amp.astype(np.uint8) # ball_amp增加了对比度的ball_cut
        highlightfilter = np.zeros((h,w), dtype="uint8")  # 加一个黑光源，去除高光
        strength = -100
        for m in range(h):
            for n in range(w):
                #计算当前点到光照中心距离(平面坐标系中两点之间的距离)
                distance = math.pow((h/2.0-n), 2) + math.pow((h/2.0-m), 2)
                if (distance < (h/2.0) **2):
                    result = (int)(strength*( 1.0 - math.sqrt(distance) / (h/2.0) ))#按照距离大小计算增强的光照值
                    f = max(0, ball_amp[n,m] + result)#判断边界 防止越界
                    highlightfilter[n,m] = np.uint8(f)
                else:
                    highlightfilter[n,m] = np.uint8(ball_amp[n,m])
        ball_amp = highlightfilter
        ball_amp[ball_amp<100]=0 #TODO
        # kernel = np.ones((4,4), np.uint8)  # TODO: change
        # ball_amp = cv2.morphologyEx(ball_amp, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀））
        # # # 画图
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
        # ax0.imshow(ball_o,cmap='gray')
        # ax0.set_title('origin image')
        # ax1.imshow(ball_amp,cmap='gray')
        # ax1.set_title('ball')
        # plt.show()
        # print(ball_o.shape) # 球尺寸
        #----------------------------------------------------------------------------------------------------------------------------------

# ------标记表面点的中心 (cv2.fitellipse)
        gray = ball_amp
        th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
        cnts, hiers = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
        points=[]# 球表面点在ball_cut中的(x,y)坐标，默认y轴向下为正
        print(len(cnts))
        for cnt in cnts:
            if len(cnt)<5:
                continue
            else:
                ellipse = cv2.fitEllipse(cnt) # ((x,y),(a,b),theta) 中心坐标，半长轴半短轴，倾斜角
                if np.max(ellipse[1]) < 35 and np.min(ellipse[1])>3:
                    points.append(np.asarray(ellipse[0]))
                    cv2.ellipse(ball_o, ellipse, (255,0, 255), 1, cv2.LINE_AA)
                    cv2.circle(ball_o, (round(ellipse[0][0]),round(ellipse[0][1])), 2, (255, 0, 0),1)  # 畫椭圆圓心
                else:
                    continue
        print(points) 
        # # 画图
        # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
        # ax0.imshow(ball_o, cmap='gray')
        # ax0.set_title('origin image')
        # ax1.imshow(ball_amp, cmap='gray')
        # ax1.set_title('ball')
        # plt.show()
        # ----------------------------------------------------------------------------------------------------------------------------------

        points = np.array(points) 
        points[:,0] -= w/2 # 把points的y坐标改成第一象限（向上为正)，圆心位于球心 
        points[:,1] -= h/2
        points[:,1] *= -1
        z = np.sqrt(r**2-points[:, 0]**2-points[:, 1]**2) #计算points的z坐标
        points = np.insert(points, 2, values=z, axis=1)  # (x,y,z), 球心为圆心的第一象限，np.array of dim n*3
        points_ps = points[~np.isnan(points).any(axis=1),:]
        f_no = np.shape(np.array(points_ps))[0]  # no. of features
        # print(points) 
        
  
# ------nomalization
        if f_no > 3:
            points_ps = points_ps[0:3, :]
            points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :]), sp.Matrix(points_ps[2, :])],
                               orthonormal=True), dtype=np.float32)
            f_no = 3
        elif f_no == 3:
            points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :]), sp.Matrix(points_ps[2, :])],
                               orthonormal=True), dtype=np.float32)
            # points_ps_n1 = spy.orthogonalize(points_ps)  # 结果一致
            # print(points_ps_n)
            # print(points_ps_n[0,0]**2+points_ps_n[0,1]**2+points_ps_n[0,2]**2,radius**2)  # (x,y,z), np.array of dim n*3
        elif f_no == 2:
            points_ps_n = np.array(
                sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :])], orthonormal=True),dtype=np.float32)
            creat_p = np.cross(np.array(points_ps_n[0]), np.array(points_ps_n[1]),axis=0) # 找第三个点
            creat_p = creat_p/np.linalg.norm(creat_p)
            creat_p.resize((1,3))
            points_ps_n = np.concatenate((points_ps_n, creat_p))#np.row_stack((points_ps_n, creat_p))
            f_no = 3
        elif f_no == 1:
            points_ps = points_ps/np.linalg.norm(points_ps)
        #     points_ps_n = np.array(points_1[:3])
        #     f_no = 3

        # print('normalized points')
        # print(points_ps_n)


# ------points matching
        if f_no == 3:
            # find the closest points
            dis = np.zeros((f_no, 6))
            for i in range(f_no):
                dis[i, :] = list(map(lambda x: np.linalg.norm(points_ps_n[i, :]-x), points_1))  # (f_no,6)距离假定点的矩阵
            # print('dis')
            # print(dis)
            index_min = np.argmin(dis, axis=1)  # fearures对应前一帧/模型中的最近的点index
            if len(index_min) == 2:
                if abs(index_min[0]-index_min[1])%3 == 0:
                    dis[1,index_min[1]]=5
                    dis[1,(index_min[1]+3)%6]=5
                    index_min = np.argmin(dis, axis=1)
            elif len(index_min) == 3:
                if abs(index_min[0]-index_min[1])%3 == 0:
                    dis[1,index_min[1]]=5
                    dis[1,index_min[2]]=5
                    dis[1,(index_min[1]+3)%6]=5
                    dis[1,(index_min[2]+3)%6]=5
                    index_min = np.argmin(dis, axis=1)
                if abs(index_min[2]-index_min[0])%3 == 0:
                    dis[2,index_min[2]]=5
                    dis[2,index_min[1]]=5
                    dis[2,(index_min[1]+3)%6]=5
                    dis[2,(index_min[2]+3)%6]=5
                    index_min = np.argmin(dis, axis=1)
                if abs(index_min[2]-index_min[1])%3 == 0:
                    dis[2,index_min[2]]=5
                    dis[2,index_min[0]]=5
                    dis[2,(index_min[2]+3)%6]=5
                    dis[2,(index_min[0]+3)%6]=5
                    index_min = np.argmin(dis, axis=1)                
            # print(index_min)
            # print('-==')
            points_co = np.array([points_1[index_min[i]] for i in range(f_no)])  # 前一帧/模型中匹配点的提取与按序排列
            # print(points_co)
            # 求旋转矩阵 及 六个点坐标  # 旋转矩阵用小数的坐标，输出用整数的
            # points_11 = np.delete(points_1, index_min, axis=0)  # 对应前一帧中的点
            # print('points_11',points_11)
            # index = [(x+3)%6 for x in range(6) if x in index_min]
            # index = np.append(index_min, index)  # 六个点对应前一帧index记录
            points_co = points_co.reshape(3,3)
            points_ps_n = points_ps_n.reshape(3,3)

            pi = sp.Matrix(points_co).T  # 转置
            pf = sp.Matrix(points_ps_n).T
            # print(pi,pf)

        elif f_no == 1:
             # find the closest points
            dis = np.zeros((f_no, 6))
            for i in range(f_no):
                dis[i, :] = list(map(lambda x: np.linalg.norm(points_ps[i, :]-x), points_1))  # (f_no,6)距离假定点的矩阵
            # print('dis')
            # print(dis)
            index_min = np.argmin(dis, axis=1)  # fearures对应前一帧/模型中的最近的点index
            points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_1[int(index_min+1)%3]), sp.Matrix(points_1[int(index_min+2)%3])],
                               orthonormal=True), dtype=np.float32)
            f_no = 3
            # find the closest points
            dis = np.zeros((f_no, 6))
            for i in range(f_no):
                dis[i, :] = list(map(lambda x: np.linalg.norm(points_ps_n[i, :]-x), points_1))  # (f_no,6)距离假定点的矩阵
            index_min = np.argmin(dis, axis=1)  # fearures对应前一帧/模型中的最近的点index
            points_co = np.array([points_1[index_min[i]] for i in range(f_no)])  # 前一帧/模型中匹配点的提取与按序排列
            pi = sp.Matrix(points_co).T  # 转置
            pf = sp.Matrix(points_ps_n).T

            




        if pi.det() != 0:  # del()不为0
            rot = pf*(pi.inv())  # 旋转矩阵
            points_ps_n1 = np.zeros(shape=(6,3))
            for i in range(len(index_min)):
                points_ps_n1[index_min[i]] = np.array(points_ps_n[i], dtype=np.float)  # 按初始顺序排列六个点坐标
                points_ps_n1[(index_min[i]+3)%6] = -np.array(points_ps_n[i], dtype=np.float)
            points_f = points_ps_n1  # np.vstack((points_ps_n,-points_ps_n))
            # print(points_f)

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
        print(c,'-------------')

        points_1 = points_f.reshape(6,3)
        print(points_f.flatten())

    store.close()
    print('--')
    print(j)
    del store, img