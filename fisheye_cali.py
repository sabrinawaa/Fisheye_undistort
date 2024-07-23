#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 17:41:26 2024

@author: sabo4ever
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the fisheye image
img = cv2.imread('WechatIMG17.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Define the dimensions of the image
height, width = img.shape[:2]
#%%

# Define the camera matrix K
# This is a simple example and might need adjustments based on your camera specifications
# K = np.array([[width / 2, 0, width / 2],
#               [0, height / 2, height / 2],
#               [0, 0, 1]], dtype=np.float32)

# fx =  [520,540,560]
# fy = fx
# for fxx in fx:
K = np.array([[770, 0, width / 2],
              [0, 520, height / 2],
              [0, 0, 1]], dtype=np.float32)

k1 = [-0.5]
k2 = [0.2]
for k11 in k1:
    for k22 in k2:
        # Define the distortion coefficients D
        # These values are example coefficients and need to be tuned to your specific setup
        D = np.array([k11, k22, 0, 0], dtype=np.float32)
        
        # Define the new camera matrix
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=1.0)
        
        '''
        The balance parameter in estimateNewCameraMatrixForUndistortRectify 
        ranges from 0 (maximally cropped) to 1 (minimally cropped). 
        Adjust it to control the trade-off between the field of view 
        and the amount of black borders in the undistorted image.
        '''
        
        # Undistort the image
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Save or display the undistorted image
        # cv2.imwrite('undistorted_image.jpg', undistorted_img)
        plt.figure()
        plt.title("k1="+str(k11)+"k2="+str(k22))
        # plt.title("fy="+str(fxx))
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        plt.savefig("fisheye_cali/k1="+str(k11)+"k2="+str(k22)+".jpg")
        # plt.savefig("fisheye_cali/fy="+str(fxx)+".jpg")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
        
 #%%       


# Define the camera matrix K
# This is a simple example and might need adjustments based on your camera specifications
# K = np.array([[width / 2, 0, width / 2],
#               [0, height / 2, height / 2],
#               [0, 0, 1]], dtype=np.float32)

# fx =  [520,540,560]
# fy = fx
# for fxx in fx:
K = np.array([[318, 0, width // 2],
              [0, 318, height // 2],
              [0, 0, 1]], dtype=np.float32)

k1 = [-1.3]
k2 = [2.1]
k3 = -0.9
k4 = 0.1

for k11 in k1:
    for k22 in k2:
        # Define the distortion coefficients D
        # These values are example coefficients and need to be tuned to your specific setup
        D = np.array([k11, k22, k3,k4], dtype=np.float32)
        
        # Define the new camera matrix
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (width, height), np.eye(3), balance=1.0)
        
        '''
        The balance parameter in estimateNewCameraMatrixForUndistortRectify 
        ranges from 0 (maximally cropped) to 1 (minimally cropped). 
        Adjust it to control the trade-off between the field of view 
        and the amount of black borders in the undistorted image.
        '''
        
        # Undistort the image
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, (width, height), cv2.CV_16SC2)
        undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        
        # Save or display the undistorted image
        # cv2.imwrite('undistorted_image.jpg', undistorted_img)
        plt.figure()
        plt.title("k1="+str(k11)+"k2="+str(k22)+"k3="+str(k3)+"k4="+str(k4))
        # plt.title("fy="+str(fxx))
        plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
        plt.savefig("fisheye_cali/k1="+str(k11)+"k2="+str(k22)+"k3="+str(k3)+"k4="+str(k4)+".jpg")
        # plt.savefig("fisheye_cali/fy="+str(fxx)+".jpg")
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #%%

   

