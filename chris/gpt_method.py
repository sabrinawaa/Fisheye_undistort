import cv2
import numpy as np
import os

# Prepare object points
obj_points = np.zeros((6*7, 3), np.float32)
obj_points[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

objpoints = []  # 3D point in real world space
imgpoints = []  # 2D points in image plane


def list_jpg_images(folder_path):
    # List to hold the image file names
    image_files = []

    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        # Check if the file is a jpg image
        if file_name.endswith('.jpg'):
            image_files.append(file_name)
    
    return image_files

folder_path = 'chessboard'  # Replace with the path to your folder
jpg_images = list_jpg_images(folder_path)
print(jpg_images)
# Load calibration images

for fname in jpg_images:
    img = cv2.imread('chessboard\{}'.format(fname))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    if ret:
        objpoints.append(obj_points)
        imgpoints.append(corners)

# Calibration
ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)


# Load the distorted image
img = cv2.imread('distorted_image.jpg')
h, w = img.shape[:2]

# Get optimal camera matrix for better undistortion
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

# Undistort
undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Crop the image
x, y, w, h = roi
undistorted_img = undistorted_img[y:y+h, x:x+w]

# Save the undistorted image
cv2.imwrite('undistorted_image.jpg', undistorted_img)


map1, map2 = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), 5)
undistorted_img = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)
