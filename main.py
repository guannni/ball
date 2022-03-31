import cv2
import func
import matplotlib.pyplot as plt
import numpy as np
from skimage import morphology, feature, transform, draw
import skimage
import trackpy as tp
import pims
import warnings
import sympy as sp


# "TODO" labels parameters to be changed.
# "todo" labels code not finished yet

warnings.filterwarnings('ignore')
frames = pims.ImageSequence(r'D:\guan2018\Thesis_exp\0417\0417\30hz1.1pp\test\*.jpg', as_grey=True)  # (r'E:\Dropbox\code\a\*.jpg', as_grey=True)
img = frames[2041]

# add a mask for the container
h, w = img.shape[:2]
r_m = 4.7 / 10. * h
mask_index = func.create_circular_mask(h, w, center=[h / 2 + 6.5, w / 2 - 2], radius=r_m - 1)
img_m = img.copy()
img_m[~mask_index] = img.mean()  # image array with mask
mask = np.zeros((h, w), np.uint8)
mask[mask_index] = 255  # mask array

# image Logarithmic change # stretch the masked image
img_g = cv2.GaussianBlur(img_m, (3, 3), 0)
img_l = func.log(43, img_g)
cv2.imshow("log", img_l)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 二进制阈值化处理 # 此处粗略，点不需要全，保证球形状完整即可
r_bl, b_bl = cv2.threshold(img_l, 150, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
# 显示图像
cv2.imshow("1", b_bl)
cv2.waitKey(0)
cv2.destroyAllWindows()

# find the biggest region
b_bool = (255 <= b_bl)
b_bool1 = ~b_bool
# b_labeled, num = skimage.measure.label(b_bool, neighbors=4, background=0, return_num=True)
plt.figure()
plt.imshow(b_bool1)
plt.title('nn')
plt.show()


b_bool = 255*b_bool

contours,hierarchy = cv2.findContours(b_bool, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0]

ellipse = cv2.fitEllipse(cnt)
im = cv2.ellipse(b_bool ,ellipse,(0,255,255),2)


plt.figure()
plt.imshow(im)
plt.title('m')
plt.show()
# max_label = 0
# max_num = 0
# for i in range( num):  # 这里从1开始，防止将背景设置为最大连通域 todo 有问题没改完
#
#     print(i)
#     if np.sum(b_labeled == i) >= max_num:
#         max_num = np.sum(b_labeled == i)
#         max_label = i
# b_br = (b_labeled == max_label)  # bool of the biggest region
# b_br = 1 * b_br  # array
# print(max_label)




#
# # hough fitting and cut the ball region
# hough_radii = np.arange(30, 50, 2)  # 半径范围
# hough_res = transform.hough_circle(b_br, hough_radii)  # 用b_br进行hough变换
# centers = []  # 保存所有圆心点坐标
# accums = []  # 累积值
# radii = []  # 半径
# for radius, h in zip(hough_radii, hough_res):
#     # 每一个半径值，取出其中两个圆
#     num_peaks = 1
#     peaks = feature.peak_local_max(h, num_peaks=num_peaks)  # 取出峰值
#     centers.extend(peaks)
#     accums.extend(h[peaks[:, 0], peaks[:, 1]])
#     radii.extend([radius] * num_peaks)
# # 画出最接近的圆
# image = np.copy(img_l)  # 切原图的复制图
# Xs = []
# Ys = []
# for idx in np.argsort(accums)[::-1][:2]:
#     center_x, center_y = centers[idx]
#     radius = radii[idx]
#     cx, cy = draw.circle_perimeter(center_y, center_x, radius+1)  # TODO: to better cut the ball, some changes are made
#     # image[cy, cx] = 255  # label the boundary of the circle
#     Xs.extend(cx)
#     Ys.extend(cy)
#
# radius += 1
# # cut the ball
# c_h = max(Ys) - min(Ys)
# c_w = max(Xs) - min(Xs)
# ball = image[min(Ys):min(Ys) + 2 * radius, min(Xs):min(Xs) + 2 * radius]
#
#
# fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
# ax0.imshow(b_br)
# ax0.set_title('origin image')
# ax1.imshow(ball)
# ax1.set_title('ball')
# plt.show()
#
#
# # adaptive threshold
# ball = cv2.medianBlur(ball, 3)
# # ret, th1 = cv2.threshold(ball, 100, 255, cv2.THRESH_BINARY)
# th2 = cv2.adaptiveThreshold(ball, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 3)  # TODO
# th3 = cv2.adaptiveThreshold(ball, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
# # titles = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
# # images = [ball, th1, th2, th3]
# # for i in [0, 1, 2, 3]:
# #     plt.subplot(2, 2, i+1), plt.imshow(images[i], 'gray')
# #     plt.title(titles[i])
# #     plt.xticks([]), plt.yticks([])
# # plt.show()
#
# ball_i = th2 | th3  # combination of 'Adaptive Mean Thresholding'&'Adaptive Gaussian Thresholding'
# # plt.imshow(ball_i)
# # plt.show()
# #
#
# # points only
# points = ball_i.copy()
# for x in range(2 * radius):
#     for y in range(2 * radius):
#         if (x - radius) ** 2 + (y - radius) ** 2 > (radius - 3.5) ** 2:  # TODO: change the radius
#             points[x, y] = 255
#
# # smooth the boundary of points
# blur = ((3, 3), 1)
# erode_ = (2, 2)
# dilate_ = (2, 2)
# p = cv2.dilate(cv2.erode(cv2.GaussianBlur(points, blur[0], blur[1]), np.ones(erode_)), np.ones(dilate_))
#
# # fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
# # ax0.imshow(points)
# # ax0.set_title('origin image')
# # ax1.imshow(p + points)
# # ax1.set_title('p')
# # plt.show()
#
# # feature the points
# f = tp.locate(p, 11, invert=True)
# # print(f)  # shows the first few rows of data
# plt.figure()  # make a new figure
# tp.annotate(f, p)  # circle the points
#
#
# points_ps = f[f.columns[0:2]].values  # (y,x), np.array of dim n*2
# points_ps -= radius
# points_ps[:,[0, 1]] = points_ps[:,[1, 0]]  # change to (x,y)
# zps = np.sqrt(radius**2-points_ps[:, 0]**2-points_ps[:, 1]**2)
# points_ps = np.insert(points_ps, 2, values=zps, axis=1)  # (x,y,z), np.array of dim n*3
# # print(points_ps)
# f_no = np.shape(np.array(f))[0]  # no. of features
# if f_no >= 3:
#     points_ps = points_ps[0:3, :]
#
# # normalizing
# if f_no <= 3:
#     points_ps_n = np.array(sp.GramSchmidt([sp.Matrix(points_ps[0,:]),sp.Matrix(points_ps[1,:]),sp.Matrix(points_ps[2,:])], orthonormal=True), dtype=np.float32)*radius
# #   print(points_ps_n[0,0]**2+points_ps_n[0,1]**2+points_ps_n[0,2]**2,radius**2)  # (x,y,z), np.array of dim n*3
# # todo else if f_no = 2:
#
# # initialization
# points_ini = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0], [0, -1, 0], [0, 0, -1]])*radius
#
# # find the closet points
# dis = np.zeros((f_no, 6))
# for i in range(f_no):
#     dis[i, :] = list(map(lambda x: np.linalg.norm(points_ps_n[i, :]-x), points_ini))  # (f_no,6)距离假定点的矩阵
# # print(dis)
# index_min = np.argmin(dis, axis=1)  # fearures对应前一帧/模型中的最近的点index
# # print(index_min)
# points_co = np.array([points_ini[index_min[i]] for i in range(f_no)])  # 前一帧/模型中匹配点的提取与按序排列
# # print(points_co)
#
#
# # 求旋转矩阵 及 六个点坐标
# if f_no >= 3:
#     points_ini1 = np.delete(points_ini, index_min, axis=0)
#     points_ini = np.vstack((points_co, points_ini1))  # 前一帧/模型中的所有点（匹配点按序排列）
#     print(points_ini)
#     index = [x for x in range(6) if x not in index_min]
#     index = np.append(index_min, index)  # 六个点初始排序index记录
#     print(index)
#     pi = sp.Matrix(points_co).T  # 转置
#     pf = sp.Matrix(points_ps_n).T
#     print(pf)
#     if pf.rank() != 0:  # 秩不为0
#         rot = pf*(pi.inv())  # 旋转矩阵
#         print(rot, rot.det())
#         points_f = np.vstack(np.hsplit(np.array(rot*sp.Matrix(points_ini).T), 2))  # 旋转后的六个点坐标
#         points_f = np.array([points_f[i] for i in index])  # 按初始顺序排列六个点坐标
#         print(points_f)
# # #   可用points_co——points_ps_n，vec——vec1验证rot
# #         vec1 = [points_ps_n[0] - points_ps_n[1], points_ps_n[1] - points_ps_n[2], points_ps_n[2] - points_ps_n[0]]
# #         vec = [points_co[0] - points_co[1], points_co[1] - points_co[2], points_co[2] - points_co[0]]
# #         print(rot*sp.Matrix(vec).T,sp.Matrix(vec1).T)
#
# # todo 求点的坐标, 重复f_no ==3
# # else if f_no == 2:
# #     b = (points_ps_n[0,0]*points_ps_n[1,2]-points_ps_n[0,2]*points_ps_n[1,0])/(points_ps_n[0,0]*points_ps_n[1,1]-points_ps_n[0,1]*points_ps_n[1,0])
# #     nor = np.array([-1*(b*points_ps_n[0,1]+points_ps_n[0,2])/points_ps_n[0,0], b, 1])
# #     nor = nor/np.linalg.norm(nor)*radius
# # # # 先求一个点