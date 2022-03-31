"""
  读取碰撞声音list -- “D:\guan2021\report_b\new\exp\2_data\audio\”,
  读取平动list -- "D:\guan2021\report_b\new\exp\2_data\frame\"，(..\frame\中文件夹数目与..\audio\中文件数目相同且一一对应）
  计算每帧对应的高度
"""
import tables as tb
import math
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
import numpy as np
import os.path
import warnings
import sympy as sp
import matplotlib
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
import h5py


warnings.filterwarnings('ignore')
# TODO: CHANGE PARAMETERS HERE------------------
fps = 240

# read position from hdf5 file
path1 = 'G:\\ball_new\\2_data\\audio\\20hz_240\\' # 读取碰撞声音list
path2 = 'G:\\ball_new\\2_data\\frame\\20hz_240_copy\\2g\\' # 读取平动list

filename = [name for name in os.listdir(path2) if name.endswith('.h5')]  # frame文件夹的名字list

ys = [i +  (i ) ** 2 for i in range(len(filename))]
colors = cm.rainbow(np.linspace(0, 1, len(ys)))
step = 1  #30

for j in range(len(filename)):  
    # 读取audio
    name = filename[j].split('.')[0]
    file_audio = path1 + filename[j]
    store_audio = h5py.File(file_audio, mode='r')
    audio = store_audio[name].value
    fps_audio = int(audio[0])
    peaks = audio - fps_audio  # unit in s

    # 读取frame
    file_frame = path2 + filename[j]
    file_frame_new = path2 + 'new\\'+ filename[j]
    store_frame = pd.HDFStore(file_frame, mode='r')
    center = store_frame.get('center').values  # center array
    matrix = store_frame.get('matrix').values
    points = store_frame.get('points').values
    store_frame.close()
    time = np.arange(0,len(center)/fps,1/fps)
    height = (1400.0/40.0*111.0)-np.sqrt(((1400.0-20.0)/40.0*111.0)**2*np.ones(len(center))-center[:,0]**2-center[:,1]**2)  # unit：pixel，球110.8pixel=40mm，容器半径1400mm，计算（容器半径-球离容器球心的纵向高度）in pixel
    frame_new = np.c_[time,center,height]
    for k in range(len(time)):
        if time[k] > peaks[0]:
            frame_new = frame_new[k:,:]
            matrix = matrix[k:,:]
            points = points[k:,:]
            break
    frame_new[:,0] = frame_new[:,0]-peaks[0]  # 把时间0点归位到第一次碰撞, [time,x,y,z](unit in [s,pixel,pixel,pixel])，x&y为相对容器画幅左下角的位置，z为相对容器底切面的位置；
    #  后面只用frame_new里的数据，别的数据时间没对齐
    peaks -= peaks[0]
    peaks = peaks.T
    print(peaks)
    peak_mark = np.zeros(len(peaks)) # 记录peak后的第一个frame帧数
    for k in range(len(peaks)):
        for l in range(len(frame_new[:,0])):
            if frame_new[l,0] < peaks[k] and frame_new[l+1,0]  > peaks[k]:
                peak_mark[k] = l+1  # peak后的第一个frame帧数 
            
    # 计算audio peak时球的位置
    peaks_p = np.zeros((len(peaks),4)) # [time,x,y,R-z] unit in [s,pixel,pixel,pixel]，x&y为相对容器画幅左下角的位置，z为相对容器底切面的位置；
    for k in range(len(peaks)):
        if k < len(peaks)-1:
            peaks_p[k,0] = peaks[k]
            peaks_p[k,1] = frame_new[int(peak_mark[k]),1] - (frame_new[int(peak_mark[k+1])-1,1]-frame_new[int(peak_mark[k]),1])*fps/(peak_mark[k+1]-peak_mark[k]-1)*(frame_new[int(peak_mark[k]),0]-peaks[k]) #todo --------------
            peaks_p[k,2] = frame_new[int(peak_mark[k]),2] - (frame_new[int(peak_mark[k+1])-1,2]-frame_new[int(peak_mark[k]),2])*fps/(peak_mark[k+1]-peak_mark[k]-1)*(frame_new[int(peak_mark[k]),0]-peaks[k])
            peaks_p[k,3] = np.sqrt(((1400.0-20.0)/40.0*111.0)**2-peaks_p[k,2]**2-peaks_p[k,1]**2)  # unit：pixel，球110.8pixel=40mm，容器半径1400mm，计算球离容器球心的纵向高度in pixel
        else:
            peaks_p[k,0] = peaks[k]
            peaks_p[k,1] = frame_new[int(peak_mark[k]),1] - (frame_new[-1,1]-frame_new[int(peak_mark[k]),1])*fps/(len(frame_new)-peak_mark[k])*(frame_new[int(peak_mark[k]),0]-peaks[k]) #todo --------------
            peaks_p[k,2] = frame_new[int(peak_mark[k]),2] - (frame_new[-1,2]-frame_new[int(peak_mark[k]),2])*fps/(len(frame_new)-peak_mark[k])*(frame_new[int(peak_mark[k]),0]-peaks[k])
            peaks_p[k,3] = np.sqrt(((1400.0-20.0)/40.0*111.0)**2-peaks_p[k,2]**2-peaks_p[k,1]**2)  
    # print(peaks_p)

    # 计算audio peak对应z方向速度
    # 最后一次碰撞算不了纵向速度，两个方案：1.删去最后一次碰撞后的frame，2.实验视频录到震台停止，认为最后一次碰撞后v_z=0
    peaks_v = np.zeros((len(peaks)-1,1))
    for k in range(len(peaks)-1):
        peaks_v[k,0] = 0.5*9.8*(peaks[k+1]-peaks[k]) + (peaks_p[k,3]-peaks_p[k+1,3])/111.0*0.04/(peaks[k+1]-peaks[k]) # unit in m/s
    # 方案1--删去最后一次碰撞后的frame：计算frame对应速度
    peaks_p = peaks_p[:-1,:]
    peaks_new = np.c_[peaks_p,peaks_v] # unit[t,pixel,pixel,m/s]

    for k in range(len(frame_new)):
        if time[k] > peaks[-1]:
            frame_new = frame_new[:k-1,:]
            matrix = matrix[:k-1,:]
            points = points[:k-1,:]
            break
    peak_f =np.zeros((len(peaks_new),3))  # peak后的第一个frame 速度 
    for k in range(len(peak_f)):
        peak_f[k,0] = (frame_new[int(peak_mark[k+1])-1,1]-frame_new[int(peak_mark[k]),1])*fps/(peak_mark[k+1]-peak_mark[k]-1)/111.0*0.04  # v_x m/s
        peak_f[k,1] = (frame_new[int(peak_mark[k+1])-1,2]-frame_new[int(peak_mark[k]),2])*fps/(peak_mark[k+1]-peak_mark[k]-1)/111.0*0.04  # v_y m/s
        peak_f[k,2] = peaks_new[k,4] - 9.8*(frame_new[int(peak_mark[k]),0]-peaks_new[k,0])  # v_z m/s
    # print(peak_f)


    frame_v = np.zeros((len(frame_new),3))  # [v_x,v_y,v_z] unit[m/s,m/s,m/s]
    for k in range(len(frame_new)):
        for l in range(len(peak_mark)-1):
            if k == peak_mark[l]:
                frame_v[k,0] = peak_f[l,0]
                frame_v[k,1] = peak_f[l,1]
                frame_v[k,2] = peak_f[l,2]
            elif k > peak_mark[l] and k < peak_mark[l+1]:
                frame_v[k,0] = peak_f[l,0]  # v_x m/s
                frame_v[k,1] = peak_f[l,1]  # v_y m/s
                frame_v[k,2] = peak_f[l,2]-9.8*(k-peak_mark[l])/fps  # v_z m/s
    # print(frame_v)
    
    frame_new_n = np.c_[frame_new,frame_v] # [time,x,y,z,v_x,v_y,v_z] in unit[s,pixel,pixel,pixel,m/s,m/s,m/s];x&y为相对容器画幅左下角的位置，z为相对容器底切面的位置；
    # print(frame_new_n)
    frame_new_n[:,1] = frame_new_n[:,1]/111.0*0.04
    frame_new_n[:,2] = frame_new_n[:,2]/111.0*0.04
    frame_new_n[:,3] = frame_new_n[:,3]/111.0*0.04  # [time,x,y,z,v_x,v_y,v_z] in unit[s,m,m,m,m/s,m/s,m/s];x&y为相对容器画幅左下角的位置，z为相对容器底切面的位置；
    print(frame_new_n)

    # 存储frame_new_n
    names = ['frame_new_n', 'points', 'matrix']
    info = tuple(list([frame_new_n, points, matrix])) # points和matrix都变成一维的了，顺序按行读
    # print(info)
    info_dict = dict(zip(names, info))
    store_frame = pd.HDFStore(file_frame_new, complib='blosc')
    # print(store_frame.keys())
    for k in range(len(frame_new_n)):
        store_frame.append(key='center_new', value=pd.DataFrame((info_dict['frame_new_n'][k],)))  # 参数输出
        store_frame.append(key='points_new', value=pd.DataFrame((info_dict['points'][k],)))
        store_frame.append(key='matrix_new', value=pd.DataFrame((info_dict['matrix'][k],)))
    store_frame.close()
    print(frame_new_n)
    print(file_frame,'-------------')









            

        
    store_audio.close()



    



