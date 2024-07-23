#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:34:19 2024

@author: sabo4ever
"""

# coding:utf-8
import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt 

#%%
# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# 棋盘格模板规格
w = 9
h = 6
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = []  # 在世界坐标系中的三维点
imgpoints = []  # 在图像平面的二维点
 
images = glob.glob('chessboard/*jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        plt.imshow(np.array(img))
        # cv2.imwrite('D:images\\grid_out.png', img)
        cv2.waitKey(1)
cv2.destroyAllWindows()

#%%
w = 10
h = 7
coords = np.array([[935,120],[932,132],[930,146],[928,159],[925,172],[923,185],[921,198],[918,212],[916,225],[914,238],
          [955,123],[952,136],[949,148],[946,162],[944,174],[941,188],[939,201],[937,214],[934,227],[932,239],
          [974,126],[970,139],[968,152],[944,162],[962,177],[959,190],[956,203],[954,216],[951,228],[948,242],
          [992,130],[988,142],[985,154],[982,167],[979,179],[976,192],[973,205],[970,218],[968,231],[956,243],
          [1009,132],[1006,145],[1002,157],[999,170],[997,182],[993,195],[990,207],[987,220],[984,232],[981,245],
          [1026,137],[1022,149],[1019,161],[1016,173],[1013,186],[1010,197],[1006,209],[1003,222],[1000,235],[998,247],
          [1042,140],[1039,152],[1036,164],[1032,176],[1028,188],[1025,200],[1022,213],[1018,225],[1015,237],[1013,249]],dtype=np.float32).reshape(70,1,2)
          
          
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w * h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objpoints = []  # Array to store object points for all images
objpoints.append(objp)

imgpoints = []  
imgpoints.append(coords)                           
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(mtx)
# print(dist)
# print(rvecs)
# print(tvecs)
#%%
# 去畸变
img2 = cv2.imread('WechatIMG17.png')
plt.imshow(img2)
#%%
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# 根据前面ROI区域裁剪图片
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
plt.imshow(np.array(dst))
# cv2.imwrite('fisheye_cali/grid_out.png', dst)
 

#%% not used now
# 反投影误差
total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("total error: ", total_error / len(objpoints))
 
# 校正视频
cap = cv2.VideoCapture('D:video\\video.mp4')
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (width, height)
video_writer = cv2.VideoWriter('D:video\\result2.mp4', cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
for frame_idx in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, frame = cap.read()
    if ret:
      image_ = cv2.undistort(frame, mtx, dist, None, newcameramtx)
      cv2.imshow('jiaozheng', image_)
      # gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
      video_writer.write(image_)
    if cv2.waitKey(10) & 0xFF== ord('q'):
        break
cap.release()
# cv2.destroyALLWindows()
 

