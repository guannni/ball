#### workflow

1. [maintry.py]跑第一帧，获得初始点坐标，换入[main1.py]

    

2. [main1.py]跑程序，输出center，六个标记点的坐标（读出时三个一组，读成6//3），旋转矩阵（读出时按行读，读成3//3）

   【读入文件：D:\guan2019\2_ball\1_pic\】

   【输出文件：D:\guan2019\2_ball\2_data\】

* [main1_trans.py]输出平动球心坐标

  

3. [main1_**.py]进行后续计算
   1. [main1_trans_v.py] 计算每组数据平动fluctuation，出traj pic；把同文件夹下数据汇总，给平动PDF，拟合；计算自相关；且可更改步长。
   
   2. [main1_trans_msd.py] 求msd
      1. [main1_msd_drawtogether.py] 把同条件下msd取均值，画到一起
      2. [main1_msd_tempslope1.py]求取均值的msd的局部斜率
      
   3. [main1_rot_v.py] 计算每组数据出traj pic；把同文件夹下数据汇总，处理获得theta(+-)-axis, euler, quaternions, displacement；出theta(+)PDF, thetaPDF, eulerPDF, displacementPDF.
   
      1. [main1_rot_autoaorre_fluc.py]计算四元数自相关&fluc.计算theta自相关&fluc
   
      2. [main1_rot_msd.py]求theta的msd
   
   --------------------------------
   
   20210913重新写了1、2步，对应文件为[maintry0913.py][main10913.py]
