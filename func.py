import numpy as np
import math


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
