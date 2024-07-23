#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 11:45:25 2024

@author: sabo4ever
"""

import cv2
import numpy as np
import math
import matplotlib.pyplot as plt


def defisheye (image,type_f):

    height, width = image.shape[:2]
    cx, cy = width // 2, height // 2  # Center of the image
    
    # Create an empty undistorted image
    undistorted_img = np.zeros_like(image)
    
    # Create meshgrid for coordinates
    x_f = (np.arange(width) - cx) 
    # norm_xf = x_f*2/width #normalised
    
    y_f = (np.arange(height) - cy)
    # norm_yf = y_f*2 /height
    
    x_grid, y_grid = np.meshgrid(x_f, y_f)
    # x_grid, y_grid = np.meshgrid(norm_xf, norm_yf) #for normalised coords
    r_f = np.sqrt(x_grid**2 + y_grid**2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        if type_f == "equidistant":
            f_h = width/ (np.pi)
            f_v = height/ np.pi
            f_eff = np.sqrt(f_h**2 + f_v**2)
            theta = r_f / f_eff
        
        elif type_f == "equisolid":
            f_h = width / (4 * np.sin(np.radians(180 / 4)))
            f_v = height / (4 * np.sin(np.radians(180 / 4)))
            f_eff = np.sqrt(f_h**2 + f_v**2)
            theta = 2 * np.arcsin(r_f / (2 * f_eff))
        elif type_f == "stereographic":
            f_h = width / (4 * np.tan(np.pi/4))
            f_v = height / (4 * np.tan(np.pi/4))
            f_eff = np.sqrt(f_h**2 + f_v**2)
            theta = 2 * np.arctan(r_f / (2 * f_eff))
            
        # elif type_f == "log": #another model, not finalised yet
        #     s = 1
        #     lamda =0.25
        #     r_p = (np.exp(r_f/s) - 1 )/lamda
        else:
            print("unknown model")
    
        r_p = f_eff * np.tan(theta) #rectilinear model
        
        
        u = cx + ((r_p/r_f) * x_grid).astype(int) #shift back to 
        v = cy + ((r_p/r_f) * y_grid).astype(int)
        
        # u = (cx + r_p * x_grid * width/2 /r_f).astype(int)
        # v = (cy + r_p * y_grid* height/2 /r_f).astype(int) #to use with normalised coordinates

    
    # Valid indices, mapping
    valid_idx = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    undistorted_img[valid_idx] = image[v[valid_idx], u[valid_idx]]
    return undistorted_img

type_f= "equisolid"

# Load the fisheye image
fisheye_image = cv2.imread('WechatIMG17.png')

undistorted_image = defisheye(fisheye_image,type_f)
# Save and display the undistorted image
cv2.imwrite('undistorted_image.jpg', undistorted_image)
plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
plt.title('Undistorted Image')
plt.show()