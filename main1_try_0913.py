# 本文件为预运行代码，用于获得初始点坐标，或者更改相关参数（见TODO）
# 确定后需要更改main10913.py和func_ball0913.py (20210913version)

import cv2
from numpy.core.shape_base import _stack_dispatcher
import func_ball0913
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
import math
 
# 计算灰度直方图
def calcGrayHist(image):
    rows, cols = image.shape
    grayHist = np.zeros([256], np.uint8)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] += 1
    return grayHist


# "TODO" labels parameters to be changed.
#  TODO：改参数还有func_ball.py要改

# 读取图片
warnings.filterwarnings('ignore')
frames = pims.ImageSequence('D:\\guan2019\\2_ball\\1_pic\\60Hz\\test0913\\0913_24\\*.jpg')#'C:\\Users\\guan\\Desktop\\*.jpg')#, as_grey=True)  # (r'E:\Dropbox\code\a\*.jpg', as_grey=True)
img = frames[52]
#--------------------------------------------------------------------------------------------------------------------------------

# 给container加mask，圈球，标圆心
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度影象
h,w = gray.shape[0], gray.shape[1] # container的尺寸 x对应height，y对应width
mask_contianer = func_ball0913.create_circular_mask(h,w, center=[w/2, h/2], radius=min(h,w)/2-4)  # TODO 加r-0.5的mask，遮挡球边阴影
gray[~mask_contianer] = 0
gray[gray>255]=255
gray = gray.astype(np.uint8) 

plt.subplot(111), plt.imshow(gray, cmap='gray')
plt.title('circle on the ball'), plt.xticks([]), plt.yticks([])
plt.show()

circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=50, minRadius=20, maxRadius=50)  # TODO 把半徑範圍縮小點，檢測內圓，瞳孔
if len(circle1[0])==1:
    x,y,r = circle1[0][0][0],circle1[0][0][1],circle1[0][0][2]
# aa=2 # hough变换总是偏一点，增加矫正参数
# x,y,r = x+aa/2,y+aa/2,r+aa/2
print(x,y,r) # 球坐标&半径
# 画图
circles = circle1[0, :, :]  # 提取為二維
circles = np.uint16(np.around(circles))  # 四捨五入，取整
img1 = img.copy()
if len(circles[:])==1:
    cv2.circle(img1, (round(x), round(y)), math.ceil(r), (255, 0, 0), 1)  # 畫圓
    cv2.circle(img1, (round(x), round(y)), 2, (255, 0, 0),1)  # 畫圓心
plt.subplot(111), plt.imshow(img1, cmap='gray')
plt.title('circle on the ball'), plt.xticks([]), plt.yticks([])
plt.show()
#----------------------------------------------------------------------------------------------------------------------------

# 提取球表面点
ball_o = img.copy()
ball_o = ball_o[math.floor(y-r):math.ceil(y+r),math.floor(x-r):math.ceil(x+r)] #切下球部分，Frame类型,
ball_amp = 1.5*ball_o # TODO 增加对比度
h,w = math.ceil(x+r)-math.floor(x-r), math.ceil(y+r)-math.floor(y-r) # ball_cut的尺寸 x对应height，y对应width
mask_ballindex = func_ball0913.create_circular_mask(w,h, center=[w/2, h/2], radius=r-0.5)  # TODO 加r-0.5的mask，遮挡球边阴影
ball_amp[~mask_ballindex] = 255 
ball_amp[ball_amp>255]=255
ball_amp[ball_amp<50]=0
ball_amp = np.round(255-ball_amp) #球黑点白(后面feature椭圆要黑底白点)
ball_amp = ball_amp.astype(np.uint8) # ball_amp增加了对比度的ball_cut
kernel = np.ones((2,2), np.uint8)  # TODO: change
ball_amp = cv2.morphologyEx(ball_amp, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀）

# 画图
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
ax0.imshow(ball_o)
ax0.set_title('origin image')
ax1.imshow(ball_amp)
ax1.set_title('ball')
plt.show()
print(ball_o.shape) # 球尺寸
#----------------------------------------------------------------------------------------------------------------------------------

# 标记表面点的中心 (cv2.fitellipse)
gray = cv2.cvtColor(ball_amp, cv2.COLOR_BGR2GRAY)
th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
cnts, hiers = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
points=[]# 球表面点在ball_cut中的(x,y)坐标，默认y轴向下为正
print(len(cnts))
for cnt in cnts:
    if len(cnt)<5:
        continue
    else:
        ellipse = cv2.fitEllipse(cnt) # ((x,y),(a,b),theta) 中心坐标，半长轴半短轴，倾斜角
        points.append(np.asarray(ellipse[0]))
        cv2.ellipse(ball_amp, ellipse, (255,0, 255), 1, cv2.LINE_AA)
        cv2.circle(ball_amp, (round(ellipse[0][0]),round(ellipse[0][1])), 2, (255, 0, 0),1)  # 畫椭圆圓心
print(points) 
# 画图
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
ax0.imshow(ball_o)
ax0.set_title('origin image')
ax1.imshow(ball_amp)
ax1.set_title('ball')
plt.show()
#----------------------------------------------------------------------------------------------------------------------------------

points = np.array(points) 
points[:,0] -= w/2 # 把points的y坐标改成第一象限（向上为正)，圆心位于球心 
points[:,1] -= h/2
points[:,1] *= -1
z = np.sqrt(r**2-points[:, 0]**2-points[:, 1]**2) #计算points的z坐标
points = np.insert(points, 2, values=z, axis=1)  # (x,y,z), 球心为圆心的第一象限，np.array of dim n*3
print(points)
f_no = np.shape(np.array(points))[0]  # no. of features
print(f_no)
if f_no >= 3:
    points = points[0:3, :]

# normalizing
if f_no == 3:
    points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points[0,:]),sp.Matrix(points[1,:]),sp.Matrix(points[2,:])], orthonormal=True), dtype=np.float32)*r
    # points_ps_n1 = spy.orthogonalize(points)*r  # 结果一致

    print(np.vstack((points_ps_n,-1*points_ps_n)))  # 输出正交化的坐标

#------------------------------------------------------------------------------

    # print(points_ps_n,points_ps_n1)
    # print(points_ps_n[0,0]**2+points_ps_n[0,1]**2+points_ps_n[0,2]**2,radius**2)  # (x,y,z), np.array of dim n*3
# todo
# elif f_no == 2:
#     points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points_ps[0, :]), sp.Matrix(points_ps[1, :])], orthonormal=True), dtype=np.float32) * r
#     print(points_ps_n)
#     creat_p = np.cross(points_ps_n[0],points_ps_n[1])
#     creat_p = creat_p/np.linalg.norm(creat_p)*radius
#     print(creat_p)
#     points_ps_n = np.row_stack((points_ps_n, creat_p))
#     f_no = 3
#     print(points_ps_n)

# # initialization
# # points_ini = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])*radius
# points_ini = np.array([[-16.56918749, -13.03656821,  27.93760754], [ 29.9007874 , -11.27742782 , 14.27454148], [  5.93862715 , 30.01398015,  16.99687334],[16.56918749, 13.03656821,  -27.93760754],[ -29.9007874 , 11.27742782 , -14.27454148], [  -5.93862715 , -30.01398015,  -16.99687334]])
# # find the closest points
# dis = np.zeros((f_no, 6))
# for i in range(f_no):
#     dis[i, :] = list(map(lambda x: np.linalg.norm(points_ps_n[i, :]-x), points_ini))  # (f_no,6)距离假定点的矩阵
# # print('---------------')
# # print(dis)
# index_min = np.argmin(dis, axis=1)  # fearures对应前一帧/模型中的最近的点index
# for i in range(f_no):
#     for j in range(i+1,f_no):
#         dis[j, index_min[i]] = 100
# index_min = np.argmin(dis, axis=1)
# # print(index_min)
# points_co = np.array([points_ini[index_min[i]] for i in range(f_no)])  # 前一帧/模型中匹配点的提取与按序排列
# # print(points_co)
# #

# # 求旋转矩阵 及 六个点坐标  # 旋转矩阵用小数的坐标，输出用整数的
# if f_no >= 3:
#     points_ini1 = np.delete(points_ini, index_min, axis=0)
#     points_ini = np.vstack((points_co, points_ini1))  # 前一帧/模型中的所有点（匹配点按序排列）
#     print(points_co)
#     index = [x for x in range(6) if x not in index_min]
#     index = np.append(index_min, index)  # 六个点初始排序index记录
#     # print(index)
#     pi = sp.Matrix(points_co).T  # 转置
#     pf = sp.Matrix(points_ps_n).T
#     if pi.det() != 0:  # 秩不为0
#         rot = pf*(pi.inv())  # 旋转矩阵
#         print('---------------')
#         print(rot, rot.det())
#         points_f = np.vstack(np.hsplit(np.array(rot*sp.Matrix(points_ini).T), 2))  # 旋转后的六个点坐标
#         points_f = np.array([points_f[i] for i in index], dtype=np.float)  # 按初始顺序排列六个点坐标
#         print(points_f)
# # #   可用points_co——points_ps_n，vec——vec1验证rot
# #         vec1 = [points_ps_n[0] - points_ps_n[1], points_ps_n[1] - points_ps_n[2], points_ps_n[2] - points_ps_n[0]]
# #         vec = [points_co[0] - points_co[1], points_co[1] - points_co[2], points_co[2] - points_co[0]]
# #         print(rot*sp.Matrix(vec).T,sp.Matrix(vec1).T)

# todo 求点的坐标, 重复f_no ==3

# # 先求一个点