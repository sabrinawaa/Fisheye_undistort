#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:58:03 2024

@author: sabo4ever
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:34:19 2024

@author: sabo4ever
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

num = str(119)
img = cv2.imread('chessboard/fisheyee-' + num +'.jpg')
new = img.copy()
#invert 
imagem = cv2.bitwise_not(img)
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 120, 0.1, 13, blockSize = 5)
corners = np.intp(corners)
for i in corners:
    # import pdb; pdb.set_trace()
    x, y = i.ravel()
    cv2.circle(imagem, (x,y),5,255,-1)
plt.imshow(imagem)
# cv2.imwrite('hough_img.png',imagem)

#%%
xx,yy = 665,538
thres = 3


for i in range(len(corners)):
    [[x,y]] = corners[i]
    if abs(xx-x)<thres and abs(yy-y)<thres:
        print(x,y)
        # cv2.circle(imagem, (x,y),5,(0,255,0),-1)
        # plt.imshow(imagem)
        idx = i
 #%% 

corners1 = np.delete(corners1,idx,axis=0) #change dynamically
corners1 = corners1.reshape(-1, 1, 2)      
#%%
corners1 = corners
if len(corners1) == 88:
    print("file saved")
    np.save('num.txt',corners1)

#%%
import cv2
import numpy as np
import matplotlib
from matplotlib.pyplot import imshow



# white color mask
img = cv2.imread(filein)
#converted = convert_hls(img)
image = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
lower = np.uint8([0, 200, 0])
upper = np.uint8([255, 255, 255])
white_mask = cv2.inRange(image, lower, upper)
# yellow color mask
lower = np.uint8([10, 0,   100])
upper = np.uint8([40, 255, 255])
yellow_mask = cv2.inRange(image, lower, upper)
# combine the mask
mask = cv2.bitwise_or(white_mask, yellow_mask)
result = img.copy()
cv2.imshow("mask",mask) 
#%%
# coding:utf-8
import cv2
import numpy as np
import glob
 
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
 
images = glob.glob('fisheye_cali/*.png')
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
        cv2.imshow('findCorners', img)
        # cv2.imwrite('D:images\\grid_out.png', img)
        cv2.waitKey(1)
cv2.destroyAllWindows()
 
# 标定
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(mtx)
# print(dist)
# print(rvecs)
# print(tvecs)
# 去畸变
img2 = cv2.imread('WechatIMG3.png')
h, w = img2.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
dst = cv2.undistort(img2, mtx, dist, None, newcameramtx)
# 根据前面ROI区域裁剪图片
# x,y,w,h = roi
# dst = dst[y:y+h, x:x+w]
cv2.imwrite('fisheye_calib/grid_out.png', dst)
 
#%%
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
 

