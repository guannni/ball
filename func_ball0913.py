import math
import cv2
import matplotlib.pyplot as plt
import numpy as np
import trackpy as tp
import sympy as sp

def create_circular_mask(h, w, center=None, radius=None):
    # create a circular mask with "h, w" = img.shape[:2]
    if center is None:  # use the middle of the image
        center = [int(w / 2), int(h / 2)]
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)

    mask = dist_from_center <= radius
    return mask


def grey_scale(image):
    # strech grey scale picture, "image" should be a np.array of a gray_pic
    img_gray = image
    rows, cols = img_gray.shape
    flat_gray = img_gray.reshape((cols * rows,)).tolist()
    A = min(flat_gray)
    B = max(flat_gray)
    # print('A = %d,B = %d' % (A, B))
    output = np.uint8(255 / (B - A) * (img_gray - A) + 0.5)
    return output


def log(c, img):
    # Logarithmic change, "c" is a constant, "img" is np.array
    output = c * np.log(1.0 + img)
    output = np.uint8(output + 0.5)
    return output


# function of rotation matrix-----------------------------------
# 不用这个，直接算就行
def normalize(a):
    a = np.array(a)
    return np.sqrt(np.sum(np.power(a, 2)))


def rot_mat3(a, b):
    # rotate from "array/list a(1*3)" to "array/list b(1*3)"
    # return a "matrix C", satisfying "B = C*np.mat(a).T" where "B = np.mat(b).T"
    rot_axis = np.cross(a, b)
    rot_angle = math.acos(np.dot(a, b) / normalize(a) / normalize(b))

    norm = normalize(rot_axis)
    rot_mat = np.zeros((3, 3), dtype="float32")

    rot_axis = (rot_axis[0] / norm, rot_axis[1] / norm, rot_axis[2] / norm)

    rot_mat[0, 0] = math.cos(rot_angle) + rot_axis[0] * rot_axis[0] * (1 - math.cos(rot_angle))
    rot_mat[0, 1] = rot_axis[0] * rot_axis[1] * (1 - math.cos(rot_angle)) - rot_axis[2] * math.sin(rot_angle)
    rot_mat[0, 2] = rot_axis[1] * math.sin(rot_angle) + rot_axis[0] * rot_axis[2] * (1 - math.cos(rot_angle))

    rot_mat[1, 0] = rot_axis[2] * math.sin(rot_angle) + rot_axis[0] * rot_axis[1] * (1 - math.cos(rot_angle))
    rot_mat[1, 1] = math.cos(rot_angle) + rot_axis[1] * rot_axis[1] * (1 - math.cos(rot_angle))
    rot_mat[1, 2] = -rot_axis[0] * math.sin(rot_angle) + rot_axis[1] * rot_axis[2] * (1 - math.cos(rot_angle))

    rot_mat[2, 0] = -rot_axis[1] * math.sin(rot_angle) + rot_axis[0] * rot_axis[2] * (1 - math.cos(rot_angle))
    rot_mat[2, 1] = rot_axis[0] * math.sin(rot_angle) + rot_axis[1] * rot_axis[2] * (1 - math.cos(rot_angle))
    rot_mat[2, 2] = math.cos(rot_angle) + rot_axis[2] * rot_axis[2] * (1 - math.cos(rot_angle))

    return np.mat(rot_mat)


# --------------------------------------------
# 结果与sp.GramSchmidt()完全一致
def qrtho3(a):
    # correct the origin 3 vec to a normalizing position
    # 'a' is an array with dim(3,3)-- (x,y,z)*3
    c = a.copy()
    for i in range(1, 3):
        k = list(map(lambda x: np.dot(c[i, :], c[x, :]) / np.dot(c[x, :], c[x, :]), range(i)))
        for j in range(3):
            for m in range(i):
                c[i, j] = c[i, j] - k[m] * c[m, j]
    return c

#----------------------------------------------------
# 本文件为预运行函数，用于获得初始点坐标
def points_ini(img):
    # 圈球，标圆心
    gray = cv2.cvtColor(2*img, cv2.COLOR_BGR2GRAY)  # 灰度影象
    circle1 = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 400, param1=100, param2=13, minRadius=30, maxRadius=50)  #把半徑範圍縮小點，檢測內圓，瞳孔
    if len(circle1[0])==1:
        x,y,r = circle1[0][0][0],circle1[0][0][1],circle1[0][0][2]

    # 提取球表面点
    ball_o = img.copy()
    ball_o = ball_o[math.floor(x-r):math.ceil(x+r),math.floor(y-r):math.ceil(y+r)] #切下球部分，Frame类型
    ball_amp = 1.5*ball_o # TODO 增加对比度
    w, h = math.ceil(x+r)-math.floor(x-r), math.ceil(y+r)-math.floor(y-r) # ball_cut的尺寸
    mask_ballindex = create_circular_mask(w,h, center=[w/2, h/2], radius=r-0.5)  # TODO 加r-0.5的mask，遮挡球边阴影
    ball_amp[~mask_ballindex] = 255 
    ball_amp[ball_amp>255]=255
    ball_amp = np.round(255-ball_amp) #球黑点白(后面feature椭圆要黑底白点)
    ball_amp = ball_amp.astype(np.uint8) # ball_amp增加了对比度的ball_cut

    # 标记表面点的中心 (cv2.fitellipse)
    gray = cv2.cvtColor(ball_amp, cv2.COLOR_BGR2GRAY)
    th, threshed = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cnts, hiers = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    points=[]# 球表面点在ball_cut中的(x,y)坐标，默认y轴向下为正
    for cnt in cnts:
        ellipse = cv2.fitEllipse(cnt) # ((x,y),(a,b),theta) 中心坐标，半长轴半短轴，倾斜角
        points.append(np.asarray(ellipse[0]))
        cv2.ellipse(ball_amp, ellipse, (255,0, 255), 1, cv2.LINE_AA)
        cv2.circle(ball_amp, (round(ellipse[0][0]),round(ellipse[0][1])), 2, (255, 0, 0),1)  # 畫椭圆圓心
    print(points) 

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
    # -----------------------------------------------------------------------------

        return np.vstack((points_ps_n,-1*points_ps_n))  # 输出正交化的坐标

    #------------------------------------------------------------------------------
