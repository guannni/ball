# 本文件为预运行代码，用于获得初始点坐标，或者更改相关参数（见TODO）
# 确定后需要更改main1.py和func_ball.py

import cv2
import func_ball
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


# "TODO" labels parameters to be changed.
#  TODO：改参数还有func_ball.py要改

warnings.filterwarnings('ignore')
frames = pims.ImageSequence('D:\\guan2019\\2_ball\\1_pic\\60Hz\\6.0\\6.0_0\\*.jpg')#, as_grey=True)  # (r'E:\Dropbox\code\a\*.jpg', as_grey=True)
img = frames[2]

# add a mask for the container
h, w = img.shape[:2]
r_m = 4.9 / 10. * h
mask_index = func_ball.create_circular_mask(h, w, center=[h / 2+2 , w / 2 -2], radius=r_m - 12)  # TODO: change the region of the stage#func.create_circular_mask(h, w, center=[h / 2 + 6.5, w / 2 - 2], radius=r_m - 1)  # TODO: change the region of the stage
img_m = img.copy()
img_m[~mask_index] = img.mean()    # image array with mask
mask = np.zeros((h, w), np.uint8)
mask[mask_index] = 0  # mask array

# image Logarithmic change # stretch the masked image
img_g = cv2.GaussianBlur(img_m, (3, 3), 0)
img_l = func_ball.grey_scale(img_m)  #  func_ball.log(43, img_g)  # log拉伸
img_l[~mask_index] = img_l.mean()    # image array with mask
# cv2.imshow("log", img_l)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 二进制阈值化处理 # 此处粗略，点不需要全，保证球形状完整即可
r_bl, b_bl = cv2.threshold(img_l, 168, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
b_bl = cv2.medianBlur(b_bl, 3) #cv2.fastNlMeansDenoising(b_bl, 10, 10, 7, 21)  #
# 显示图像
cv2.imshow("1", b_bl)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # 找最大连通域（不好用
# b_bool = (r_bl < b_bl)
# b_labeled, num = skimage.measure.label(b_bool, neighbors=8, background=0, return_num=True)
# # find the biggest region
# max_label = 0
# max_num = 0
# for i in range(num):  # 这里从1开始，防止将背景设置为最大连通域
#     if np.sum(b_labeled == i) >= max_num:
#         max_num = np.sum(b_labeled == i)
#         max_label = i
# b_br = (b_labeled == max_label)  # bool of the biggest region
# b_br = 255 - 255 * b_br  # array
# img_l = b_br
#
kernel = np.ones((4,4), np.uint8)  # TODO: change
img_l = cv2.morphologyEx(b_bl, cv2.MORPH_OPEN, kernel)  # 开运算（先腐蚀，再膨胀）


plt.figure()
plt.imshow(img_l, cmap=plt.cm.gray)
plt.title('the biggest region')
plt.show()

b_br = img_l


# cut the ball
b_br = np.clip(b_br, 0, 255)  # 归一化也行
b_br = np.array(b_br, np.uint8)
contours, hierarchy = cv2.findContours(b_br, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0]
x, y, w1, h1=cv2.boundingRect(cnt)
print(x,y,w1,h1)  # region of ball
# 框出球
b_br1 = cv2.rectangle(b_br, (x-4, y-2), (x+w1+2, y+h1+2), (255, 255, 255), 1)  # 框出球 # TODO 更改框的起点终点

plt.figure()
plt.imshow(b_br1, cmap=plt.cm.gray)
plt.title('rect')
plt.show()

r = int(max(h1, w1)/2)

# ball = b_br[y - 1:(y + 2*r + 1), x - 3:(x + 2*r -1)]  # 注意(y,x) # 二值球  TODO 根据框的位置改
# ball_i = img[y - 1:(y + 2*r + 1), x - 3:(x + 2*r -1)]  # 原球  # TODO 根据框的位置改
ball = b_br[y - 2:(y + 2*r + 2), x - 3:(x + 2*r+1 )]  # 注意(y,x) # 二值球  TODO 根据框的位置改
ball_i = img[y - 2:(y + 2*r + 2), x - 3:(x + 2*r+1 )]  # 原球  # TODO 根据框的位置改

fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
ax0.imshow(ball_i, cmap=plt.cm.gray)
ax0.set_title('origin image')
ax1.imshow(ball, cmap=plt.cm.gray)
ax1.set_title('binary ball')
plt.show()


#----求ball二值图法一：重新处理
# adaptive threshold
# ball_i = cv2.medianBlur(ball_i, 3)
# th2 = cv2.adaptiveThreshold(ball_i, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 4)  # TODO
# th3 = cv2.adaptiveThreshold(ball_i, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)
#
# ball_i = th2 | th3  # combination of 'Adaptive Mean Thresholding'&'Adaptive Gaussian Thresholding'
# plt.imshow(ball_i)
# plt.show()

#----求ball二值图法二：承上
# ball_i = ball

# 给球加mask
points = ball.copy()
ballmask = np.zeros((2*r+1, 2*r+1), np.uint8)
mask_ballindex = func_ball.create_circular_mask(2*r + 4, 2*r + 4, center=[r+3, r+1], radius=r-1) #center=[r+1,r+1], radius=r-2  # TODO
points[~mask_ballindex] = 255  # ball_m[mask_ballindex].mean()  # mask array

# cv2.imshow("ball_m", points)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# smooth the boundary of points
blur = ((3, 3), 1)
erode_ = (3,3)
dilate_ = (3,3)
p = cv2.erode(cv2.dilate(cv2.GaussianBlur(points, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))


# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
# ax0.imshow(ball_i)
# ax0.set_title('origin image')
# ax1.imshow(p + points)
# ax1.set_title('p')
# plt.show()

# feature the points
f = tp.locate(p, 11, invert=True) # 11
# print(f)  # shows the first few rows of data
plt.figure()  # make a new figure
tp.annotate(f,  ball_i)  #p)#   # circle the points


points_ps = f[f.columns[0:2]].values  # (y,x), np.array of dim n*2
radius = r
points_ps -= radius
points_ps[:,[0, 1]] = points_ps[:,[1, 0]]  # change to (x,y)
zps = np.sqrt(radius**2-points_ps[:, 0]**2-points_ps[:, 1]**2)
points_ps = np.insert(points_ps, 2, values=zps, axis=1)  # (x,y,z), np.array of dim n*3
print(points_ps)
f_no = np.shape(np.array(f))[0]  # no. of features
print(f_no)
if f_no >= 3:
    points_ps = points_ps[0:3, :]

# normalizing
if f_no == 3:
    points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points_ps[0,:]),sp.Matrix(points_ps[1,:]),sp.Matrix(points_ps[2,:])], orthonormal=True), dtype=np.float32)*r
    # points_ps_n1 = spy.orthogonalize(points_ps)*radius  # 结果一致

# -----------------------------------------------------------------------------

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