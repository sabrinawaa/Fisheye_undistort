#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 14:09:48 2024

@author: sabo4ever
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
#%%
def cylindrical_to_cartesian(img, fov_horizontal=180, fov_vertical=85, k1 = 0.25, fx=1300, fy=150):
    height, width = img.shape[:2]
    fov_horizontal_rad = np.radians(fov_horizontal)
    fov_vertical_rad = np.radians(fov_vertical)
    
    # Calculate the radius of the cylindrical projection
    R = width / fov_horizontal_rad #or tan check
    
    # Calculate the height scaling factor
    H = height /np.tan(fov_vertical_rad / 2) #or sin or tan
    # print((R,H))
    
    new_img = np.zeros_like(img)
    
    for y in range(height):
        for x in range(width):
            # print("x=",x,"y=",y)
            theta = (x - width / 2) / R
            
            h = (y - height / 2) / H
            # print(theta,h)
            
            X = R * np.tan(theta)
            Y = h * np.sqrt(X**2 + R**2) 
            
            X_norm = X/fx
            Y_norm = Y/fy
            R2 = X_norm**2 + Y_norm**2
            # Apply radial distortion correction
            r2 = (X**2 + Y**2) / (fx**2 +fy**2)
            radial_distortion = 1 + k1 * r2
            
            
            X_distorted = X_norm * radial_distortion
            Y_distorted = Y_norm * radial_distortion
            
            new_x = int(X + width / 2 )
            new_y = int(Y + height / 2)
            
            # print(new_x,new_y)
            
            if 0 <= new_x < width and 0 <= new_y < height:
                new_img[new_y, new_x] = img[y, x]
    
    # Use interpolation to fill the gaps
    mask = (new_img.sum(axis=2) == 0).astype(np.uint8) * 255
    new_img = cv2.inpaint(new_img, mask, 3, cv2.INPAINT_TELEA)
    
    return new_img

# Load the panorama image
img = cv2.imread("../WechatIMG1.png")  # Update with your image path

# Apply the inverse cylindrical transformation
rectified_img = cylindrical_to_cartesian(img)

# Display the original and rectified images
plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.title("Original Panorama Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Rectified Image")
plt.imshow(cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.imsave("rec_method.png", rectified_img)
plt.show()

#%%
h, w = rectified_img.shape[:2]
mtx = np.array([[1300, 0.00000000e+00, w//2],
        [0.00000000e+00, 150, h//2],
        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist = np.array([[0.25,  -0.0,  -0.0 , -0.0, 0.1]])

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))  # 自由比例参数
dst = cv2.undistort(rectified_img, mtx, dist, None, newcameramtx)
plt.title("dist="+str(dist) + 'fx,fy=' +str(mtx[0][0]) + ',' + str(mtx[1][1]))
plt.figure()
plt.imshow(dst)
cv2.imwrite('undistorted.png', dst)

#%% test meshgrid
img = cv2.imread("../WechatIMG1.png") 
fov_horizontal=180
fov_vertical=85

height, width = img.shape[:2]
fov_horizontal_rad = np.radians(fov_horizontal)
fov_vertical_rad = np.radians(fov_vertical)

# Calculate the radius of the cylindrical projection
R = width / fov_horizontal_rad 

# Calculate the height scaling factor
H = height /np.tan(fov_vertical_rad / 2) #or sin or tan

new_img = np.zeros_like(img)

y = np.arange(height)
x = np.arange(width)

# y=np.array([344])
# x=np.array([298,299,300])


x_grid, y_grid = np.meshgrid(x,y)

theta = (x_grid - width / 2) / R
h = (y_grid - height / 2) / H

X = R * np.tan(theta)
Y = h * np.sqrt(X**2 + R**2)

new_x = (X + width / 2).astype(int)
new_y = (Y + height / 2).astype(int)


valid_idx = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
new_img[new_y[valid_idx], new_x[valid_idx]] = img[y_grid[valid_idx], x_grid[valid_idx]]

# # Use interpolation to fill the gaps
# mask = (new_img.sum(axis=2) == 0).astype(np.uint8) * 255
# new_img = cv2.inpaint(new_img, mask, 3, cv2.INPAINT_TELEA)

# map_x = new_x.astype(np.float32)
# map_y = new_y.astype(np.float32)

# Use cv2.remap for faster interpolation
# new_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    

plt.title("Rectified Image")
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
    


#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def cylindrical_to_cartesian(img, fov_horizontal=180, fov_vertical=85):
    height, width = img.shape[:2]
    fov_horizontal_rad = np.radians(fov_horizontal)
    fov_vertical_rad = np.radians(fov_vertical)
    
    # Calculate the radius of the cylindrical projection
    R = width / fov_horizontal_rad
    
    # Calculate the height scaling factor
    H = height / np.tan(fov_vertical_rad / 2)
    
    new_img = np.zeros_like(img)
    
    y = np.arange(height)
    x = np.arange(width)
    
    x_grid, y_grid = np.meshgrid(x, y)
    
    theta = (x_grid - width / 2) / R
    h = (y_grid - height / 2) / H
    
    X = R * np.tan(theta)
    Y = h * np.sqrt(X**2 + R**2)
    
    new_x = (X + width / 2).astype(int)
    new_y = (Y + height / 2).astype(int)
    
    valid_idx = (new_x >= 0) & (new_x < width) & (new_y >= 0) & (new_y < height)
    new_img[y_grid[valid_idx], x_grid[valid_idx]] = img[new_y[valid_idx], new_x[valid_idx]]

    # Use interpolation to fill the gaps
    mask = (new_img.sum(axis=2) == 0).astype(np.uint8) * 255
    new_img = cv2.inpaint(new_img, mask, 3, cv2.INPAINT_TELEA)
    
    return new_img

# Load the panorama image
img = cv2.imread("../WechatIMG17.png")  # Update with your image path

# Apply the inverse cylindrical transformation
rectified_img = cylindrical_to_cartesian(img)

# Display the original and rectified images
plt.figure(figsize=(15, 10))

plt.subplot(1, 2, 1)
plt.title("Original Panorama Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Rectified Image")
plt.imshow(cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

#%% with radial distort
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def cylindrical_to_cartesian_with_undistort(img, fov_horizontal=180, fov_vertical=85, mtx=None, dist=None):
    height, width = img.shape[:2]
    fov_horizontal_rad = np.radians(fov_horizontal)
    fov_vertical_rad = np.radians(fov_vertical)
    
    # Calculate the radius of the cylindrical projection
    R = width / fov_horizontal_rad
    
    # Calculate the height scaling factor
    H = height / (2 * np.tan(fov_vertical_rad / 2))
    
    # Extract intrinsic camera parameters
    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    
    # Distortion coefficients
    k1 = dist[0, 0]
    
    # Create meshgrid for coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)
    
    # Calculate theta and h for cylindrical coordinates
    theta = (x_grid - width / 2) / R
    h = (y_grid - height / 2) / H
    
    # Map back to Cartesian coordinates
    X = R * np.tan(theta)
    Y = h * np.sqrt(X**2 + R**2)
    
    # Apply radial distortion correction
    r2 = (X**2 + Y**2) / (fx**2 + fy**2)
    radial_distortion = 1 + k1 * r2
    X_distorted = X * radial_distortion
    Y_distorted = Y * radial_distortion
    
    # Map distorted coordinates back to image plane using intrinsic parameters
    new_x = (fx * (X_distorted / R) + cx).astype(np.int32)
    new_y = (fy * (Y_distorted / R) + cy).astype(np.int32)
    
    # Ensure new coordinates are within bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    # Create new image using the transformed coordinates
    new_img = np.zeros_like(img)
    new_img[new_y, new_x] = img[y_grid, x_grid]
    
    # Use interpolation to fill the gaps
    mask = (new_img.sum(axis=2) == 0).astype(np.uint8) * 255
    new_img = cv2.inpaint(new_img, mask, 3, cv2.INPAINT_TELEA)
    
    return new_img

# Load the panorama image
img = cv2.imread("../WechatIMG17.png")  # Update with your image path

# Camera matrix and distortion coefficients
h, w = img.shape[:2]
mtx = np.array([[1300, 0.0, w // 2],
                [0.0, 150, h // 2],
                [0.0, 0.0, 1.0]])
dist = np.array([[0.25, 0.0, 0.0, 0.0, 0.0]])

# Apply the inverse cylindrical transformation with undistortion
rectified_img = cylindrical_to_cartesian_with_undistort(img, mtx=mtx, dist=dist)

# Display the original and rectified images
plt.figure(figsize=(20, 10))

plt.subplot(1, 2, 1)
plt.title("Original Panorama Image")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Rectified and Undistorted Image")
plt.imshow(cv2.cvtColor(rectified_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.show()

#%% use remap
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Load the panorama image
img = cv2.imread("../WechatIMG17.png")

fov_horizontal = 180
fov_vertical = 85
fx,fy = 1300,150

k1 = 0.0
x_scale = 2.4
y_scale = 0.75

height, width = img.shape[:2]
fx,fy  = width,150
fov_horizontal_rad = np.radians(fov_horizontal)
fov_vertical_rad = np.radians(fov_vertical)

K = np.array([[fx, 0, width // 2],
              [0, fy, height // 2],
              [0, 0, 1]], dtype=np.float32)

D = np.array([k1, 0, 0, 0], dtype=np.float32)

# Calculate the radius of the cylindrical projection
R = width / fov_horizontal_rad

# Calculate the height scaling factor
H = height / np.tan(fov_vertical_rad / 2)

# Create meshgrid for coordinates
x = np.arange(width)
y = np.arange(height)
x_grid, y_grid = np.meshgrid(x, y)



y_c = (y_grid - height/2) * y_scale
x_c = (x_grid - width/2) * x_scale

yy = y_c * H / np.sqrt(x_c**2 + R**2)
xx = R * np.arctan(x_c/R)

X_norm = xx/fx
Y_norm = yy/fy
R2 = X_norm**2 + Y_norm**2
# Apply radial distortion correction
radial_distortion = 1 + k1 * R2
X_distorted = X_norm * radial_distortion
Y_distorted = Y_norm * radial_distortion


# new_x = (xx  + width / 2).astype(np.float32)
# new_y = (yy + height / 2).astype(np.float32)

new_x = (fx * X_distorted + width / 2).astype(np.float32)
new_y = (fy * Y_distorted + height / 2).astype(np.float32)

new_x = np.clip(new_x, 0, width - 1)
new_y = np.clip(new_y, 0, height - 1)

# Use cv2.remap for faster interpolation
new_img = cv2.remap(img, new_x, new_y, interpolation=cv2.INTER_LINEAR, 
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, D, (width, height), 0, (width, height))
# map1, map2 = cv2.initUndistortRectifyMap(K, D, None, newcameramtx, (width, height), cv2.CV_32FC1)
# map_x_combined = map1[new_y.astype(int), new_x.astype(int)] # not right
# map_y_combined = map2[new_y.astype(int), new_x.astype(int)]

# # new_img2 = cv2.remap(new_img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
# final_img = cv2.remap(img, map_x_combined, map_y_combined, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

# Display the original and rectified images
plt.figure()
plt.title("Rectified Image")
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
# plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
plt.axis('off')
# plt.imsave("rectified.png", cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

#%%
def cylindrical_projection(image,fov_horizontal=180,fov_vertical=85):
    h, w = image.shape[:2]
    fov_horizontal_rad = np.radians(fov_horizontal)
    fov_vertical_rad = np.radians(fov_vertical)
    fx = w/fov_horizontal_rad
    fy = h/fov_vertical_rad

    # Define the cylindrical coordinates
    x_c = w / 2
    y_c = h / 2
    
    x_p = np.arange(w)
    y_p = np.arange(h)
    x_p_grid, y_p_grid = np.meshgrid(x_p, y_p)

    # Reverse mapping equations
    X = (x_p_grid - x_c) / fx
    Y = (y_p_grid - y_c) / fy

    theta = np.arctan(X)
    h_ = Y * np.sqrt(1 + X**2)
    
    x_grid = fx * theta + x_c
    y_grid = fy * h_ + y_c

    # Remap the image to correct the cylindrical projection
    corrected_image = cv2.remap(image, x_grid.astype(np.float32), y_grid.astype(np.float32), 
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return corrected_image

new_img = cylindrical_projection(img)
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))

#%%
def spherical_projection_reverse_map(image,fov_horizontal=180,fov_vertical=85 ):
    height, width = image.shape[:2]
    new_width = int(width *2.2)
    
    fov_horizontal_rad = np.radians(fov_horizontal)
    fov_vertical_rad = np.radians(fov_vertical)
    fx = width/fov_horizontal_rad
    fy = height/fov_vertical_rad
    
    x_c = new_width // 2
    y_c = height // 2

    # Create a meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(new_width), np.arange(height))
    
    # Convert to normalized spherical coordinates
    x_norm = (x - x_c) / fx
    y_norm = (y - y_c) / fy
    
    # Convert normalized coordinates to spherical coordinates
    theta = np.arctan(x_norm)
    phi = np.arcsin(y_norm / np.sqrt(x_norm**2 + 1))

    # Map spherical coordinates back to image plane
    x_map = fx * theta + width//2
    y_map = fy * phi + y_c
    
    # x_map = x_map +0.5 * (y_map-width//2)

    # Remap the image using the spherical projection reverse maps
    
    
    return x_map.astype(np.float32),y_map.astype(np.float32)

x_map,y_map = spherical_projection_reverse_map(dst)
time1 = datetime.now()
new_img = cv2.remap(dst, x_map, y_map, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
time2 = datetime.now()
print(time2-time1)
plt.figure()
plt.imshow(cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
