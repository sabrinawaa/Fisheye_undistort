#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 10:44:37 2024

@author: sabo4ever
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import map_coordinates
from datetime import datetime
#%%

def undistort_map (img, x_scale, y_scale, pan_angle,fov_h=180, fov_v=85):
    pi = 3.14159265
    height, width = img.shape[:2]
    new_width = int(width *2.2)
    
    fov_horizontal_rad = fov_horizontal * pi/180
    fov_vertical_rad = fov_vertical * pi/180
    
    # Calculate the radius of the cylindrical projection
    fx = width / fov_horizontal_rad
    
    # Calculate the height scaling factor
    fy = height / fov_vertical_rad # or tan
    
    # Calculate the corresponding horizontal shift based on the pan angle
    shift_x = width * np.tan(np.radians(pan_angle)) /2
    shifts = np.linspace(0, shift_x * 2, height)
    factors = (width + shifts)/(width)
    
    # Create meshgrid for coordinates
    x = np.arange(new_width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    
    y_c = (y_grid - height/2) * y_scale
    x_c = (x_grid - new_width/2) * x_scale
    
    x_c = x_c / factors[:, np.newaxis]
        
    yy = fy * y_c / fx / np.sqrt((x_c/fx)**2 + 1)
    xx = fx * np.arctan(x_c/fx)
    
    new_x = xx + width//2
    new_y = yy + height//2
    
    return new_x.astype(np.float32), new_y.astype(np.float32)

img = cv2.imread("../WechatIMG1.png")
# img = cv2.imread("frame.png")
fov_horizontal = 180
fov_vertical = 85
pan_angle = 25

x_scale = 2
y_scale = 1

new_x, new_y = undistort_map(img, x_scale, y_scale, pan_angle, fov_horizontal,fov_vertical)

time1 = datetime.now()

new_img = cv2.remap(img, new_x, new_y, interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


time2 = datetime.now()

plt.figure()
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

print(time2-time1)

#%%


kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])  # Simple sharpening kernel
sharpened_image = cv2.filter2D(new_img, -1, kernel)

def sharpen_radial(img):
    # Calculate the radial distance map
    height, width = img.shape[:2]
    center_x, center_y = width / 2, height / 2
    y, x = np.indices((height, width))
    radial_distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    radial_distance = radial_distance / np.max(radial_distance)
    
    # Sharpening kernel
    sharpening_kernel = np.array([[0, -1, 0], 
                                  [-1, 5, -1], 
                                  [0, -1, 0]])
    
    # Apply the sharpening filter to the remapped image
    sharpened_img = cv2.filter2D(img, -1, sharpening_kernel)
    
    # Create a weighted mask based on the radial distance
    mask = radial_distance**2  # Closer to the edge gets more weight
    mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    
    # Combine the original and sharpened images using the mask
    combined_img = img.astype(np.float32) * (1 - mask) + sharpened_img.astype(np.float32) * mask
    combined_img = np.clip(combined_img, 0, 255).astype(np.uint8)
    
    return combined_img
# Display the original and rectified images

sharpened_image = sharpen_radial(new_img)

#%%
def remap_with_spline(img, map_x, map_y, output_shape):
    # Ensure the coordinates are in the correct shape and type for scipy
    
    # Create an empty array for the remapped image with the size of the new dimensions
    remapped_img = np.empty((output_shape[0], output_shape[1], img.shape[2]), dtype=img.dtype)
    
    # Apply the spline interpolation for each color channel
    for i in range(img.shape[2]):  # Iterate over the color channels
        remapped_img[..., i] = map_coordinates(img[..., i], 
                                               [map_y, map_x], 
                                               order=3, mode='nearest')
    
    return remapped_img

remapped_img = remap_with_spline(img, new_x, new_y, new_img.shape).reshape(new_img.shape)

plt.subplot(1, 2, 1)
plt.title("Rectified Image")
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Sharpened Image")
plt.imshow(cv2.cvtColor(remapped_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
# plt.imsave("sharpened.png", cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2RGB))
plt.show()