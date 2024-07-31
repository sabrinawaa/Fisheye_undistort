# write the distortion process as a function to minimise

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import minimize

# Load the fisheye image
img = cv2.imread('chris\\bkg-119.jpg')
# plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
# plt.show()


# 去畸变
height, width = img.shape[:2]

mtx_ini = np.array([[width, 0, width / 2],
              [0, height, height / 2],
              [0, 0, 1]])

dist_ini = np.array([[-0.46242643, -1.41310908,  0.0, -0.0,  1.94701982]])



iterations = 0
def undistort(guess):
    
    
    a, b, c, d, e, f, g, h, i, j, k, l, m, n = guess
    global iterations
    iterations += 1
    


    ##--------# undistort the image #--------##
    mtx = np.array([[a, b, c],
                    [d, e, f],
                    [g, h, i]])
    dist = np.array([[j, k, l, m, n]])
    # newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))  # 自由比例参数
    # dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(mtx, dist, (width, height), np.eye(3), balance=1.0)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(mtx, dist, np.eye(3), new_K, (width, height), cv2.CV_16SC2) 
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    
    
    ##--------# save the undistorted image #--------##
    path = r'C:\Users\yx200\Desktop\Fisheye_undistort\chris\trial_{}'.format(iterations)
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + r'\trial_{}.jpg'.format(iterations))
    plt.close()
    with open (path + r'\trial_{}.txt'.format(iterations), 'w') as file:
        file.write(str(mtx)+'\n')
        file.write(str(dist))
        
        
        
    ##--------# finding the chessboard #--------##
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply adaptive thresholding to enhance the chessboard pattern
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Detect chessboard corners
    ret, corners = cv2.findChessboardCorners(thresh, (8, 8), None)

    if ret:
        # Draw chessboard corners
        print('Chessboard detected')
        dst = cv2.drawChessboardCorners(dst, (8, 8), corners, ret)

    plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
    plt.savefig(path + r'\chess_{}.jpg'.format(iterations))
    plt.close()
    
    
    
    return 2# minimised value
    
# cv2.imwrite('undistorted.png', dst)

ini_guess = np.array([*mtx_ini.flatten(), *dist_ini.flatten()])
result = minimize(undistort, ini_guess)
minimized_params = result.x
print(minimized_params)