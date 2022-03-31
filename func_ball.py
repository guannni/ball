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
    # add a mask for the container
    h, w = img.shape[:2]
    r_m = 4.9 / 10. * h
    mask_index = create_circular_mask(h, w, center=[h / 2 , w / 2 ], radius=r_m - 1)  # TODO: change the region of the stage#func.create_circular_mask(h, w, center=[h / 2 + 6.5, w / 2 - 2], radius=r_m - 1)  # TODO: change the region of the stage
    img_m = img.copy()
    img_m[~mask_index] = img.mean()    # image array with mask
    mask = np.zeros((h, w), np.uint8)
    mask[mask_index] = 0  # mask array

    # stretch the masked image
    img_g = cv2.GaussianBlur(img_m, (3, 3), 0)
    img_l = grey_scale(img_m)  #  func_ball.log(43, img_g)  # log拉伸
    img_l[~mask_index] = img_l.mean()    # image array with mask

    # 二进制阈值化处理 # 此处粗略，点不需要全，保证球形状完整即可
    r_bl, b_bl = cv2.threshold(img_l, 100, 255, cv2.THRESH_BINARY)  # TODO: change the lower boundary
    b_bl = cv2.fastNlMeansDenoising(b_bl, 10, 10, 7, 21)  # cv2.medianBlur(b_bl, 3) #

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
    mask_ballindex = create_circular_mask(2*r + 3, 2*r + 3, center=[r, r], radius=r-2) #r-2  # TODO
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
