# main2--处理convex容器乒乓球系列数据
# 批量处理一个路径下的.wav文件，将[fps,peak1,peak2,...]以.wav文件名为key存到同一个.h5文件下 *h5中fps和peak的输出顺序要查
import numpy as np
import math
import matplotlib.pyplot as plt
import librosa
import librosa.display as ld
from scipy.signal import find_peaks
import os.path
import glob
import pandas as pd
import wave
import contextlib
import h5py

def traversalDir_FirstDir(path):  # 返回一级子文件夹名字
    list = []
    if (os.path.exists(path)):    #获取该目录下的所有文件或文件夹目录路径
        files = glob.glob(path + '\\*' )
        # print(files)
        for file in files:            #判断该路径下是否是文件夹
            h = os.path.split(file)
            list.append(h[1])
            # if (os.path.isdir(file)):                #分成路径和文件的二元元组
            #     h = os.path.split(file)
            #     # print(h[1] )
            #     list.append(h[1])
        return list

name = '40hz6g_2'  # todo
path1 = 'g:\\ball_new\\video\\40hz_240\\'+ name +'.m4a'  # audio文件夹
path2 = 'G:\\ball_new\\2_data\\audio\\40hz_240\\'+ name +'.h5'  # 输出到一个文件里
store = h5py.File(path2,'w')

x , sr = librosa.load(path1)
print(type(x), type(sr))
print(x.shape, sr)
# plt.figure(figsize=(14, 5))
# librosa.display.waveplot(x, sr=sr)
# plt.show()

rate = 22050 #default
start_peak = 103956#106635(40hz6g_1)#202020(40hz4g_3)#138664(40hz4g_2)#132572(40hz4g_1) #135820(40hz2g_3) #166629(40hz2g_2) # 303674(40hz2g_1) 
#122027(20hz8g_3) #104925(20hz8g_2) #172737(20hz8g_1)#156228(20hz6g_2) #143226(20hz6g_1) #213477(20hz2g_2) #355999(20hz2g_1) #200086 (20hz4g_3)  #116326 (20hz4g_2)#141056(20hz4g_1) # 0   # todo
end_peak = 26489392#26566402(40hz6g_1)#26393489(40hz4g_3)#26504225(40hz4g_2)#26602900(40hz4g_1)  #26816806(40hz2g_3) #29067220(40hz2g_2) #31328755(40hz2g_1) 
#13564655(20hz8g_3)  #13552841(20hz8g_2)#14677423(20hz8g_1) #27804519(20hz6g_2) #27428447(20hz6g_1) #27189551(20hz2g_2) #28301320(20hz2g_1) #28038666 (20hz4g_3)#27407571 (20hz4g_2)#7116169 (20hz4g_1)#len(x)   # todo
peaks, _ = find_peaks(x, distance=500)
peaks_new = peaks[x[peaks]>np.average(x[peaks])+0.02]  # todo 0.05 (20hz4g)

x=x[start_peak+1:end_peak]
peaks_new = peaks_new[peaks_new<end_peak]
peaks_new = peaks_new - start_peak -1
peaks_new = peaks_new[peaks_new>0]
plt.plot(peaks_new, x[peaks_new], "xr"); plt.plot(x); plt.legend(['distance'])
plt.show()

t = rate+peaks_new/rate 
print(peaks_new,peaks_new/rate,t)


store.create_dataset(name,data=t) # [rate+pk1,rate+pk2,....] unit:rate(fps), pk(s)
store.close()

