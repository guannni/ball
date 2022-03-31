# 图片位深8to24
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

import sys
import shutil

path='D:\\guan2019\\2_ball\\1_pic\\60Hz\\test0913\\0913\\' # '/home/lguan/'
newpath='D:\\guan2019\\2_ball\\1_pic\\60Hz\\test0913\\0913_24\\'
def turnto24(path):
    files = os.listdir(path)
    files = np.sort(files)
    i=0
    for f in files:
        imgpath = path + f
        img=Image.open(imgpath).convert('RGB')
        dirpath = newpath 
        file_name, file_extend = os.path.splitext(f)
        dst = os.path.join(os.path.abspath(dirpath), file_name + '.jpg')
        img.save(dst)

turnto24(path)
