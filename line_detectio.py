#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 17:03:19 2024

@author: sabo4ever
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
img = cv2.imread('WechatIMG17.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Edge detection
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)


plt.imshow(edges)
#%%
# convert to gray
img = cv2.imread('WechatIMG17.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# threshold
thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

# morphology edgeout = dilated_mask - mask
# morphology dilate
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
dilate = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, kernel)

# get absolute difference between dilate and thresh
diff = cv2.absdiff(dilate, thresh)

# invert
edges = 255 - diff

# write result to disk
# cv2.imwrite("cartoon_thresh.jpg", thresh)
# cv2.imwrite("cartoon_dilate.jpg", dilate)
# cv2.imwrite("cartoon_diff.jpg", diff)
cv2.imwrite("edges.jpg", edges)

# display it
# cv2.imshow("thresh", thresh)
# cv2.imshow("dilate", dilate)
# cv2.imshow("diff", diff)
plt.imshow(edges)

#%%
# Hough Line Transform to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

# Draw the detected lines on the image
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(edges, (x1, y1), (x2, y2), (0, 0, 255), 2)

plt.imshow(edges)