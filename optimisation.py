import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the image
# image_path = '../WechatIMG17.png'
image_path = 'chessboard/bkg-115.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#%%

# Placeholder grid points (these should be your manually marked points)
# Example: image_points should be in pixel coordinates
image_points = np.load("gridpoints/corners115.npy").astype(np.float32)

# Corresponding ideal grid points in undistorted space
# These points form a regular grid
y = np.linspace(336,518,8)
x = np.linspace(780,1168,11)
world_points = np.meshgrid(x,y)
world_points = np.vstack([world_points[0].ravel(), world_points[1].ravel()]).T.reshape(-1,1,2)

#%%
# Initial guess for distortion coefficients (k1, k2, p1, p2, k3) and focal lengths (fx, fy)
initial_guess = np.array([0.0, 0.0, 0.0, 0.0, 0.0, image.shape[1], image.shape[0]])

# Define the optimization function
def distortion_error(params, image_points, world_points, image_size):
    k1, k2, p1, p2, k3, fx, fy = params
    camera_matrix = np.array([[fx, 0, image_size[1]/ 2],
                              [0, fy, image_size[0] / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    
    undistorted_points = cv2.undistortPoints(image_points, camera_matrix, dist_coeffs, P=camera_matrix)
    
    # Calculate the error as the difference between undistorted points and world points
    error = np.linalg.norm(undistorted_points - world_points, axis=1)
    return np.sum(error)

# Perform the optimization
image_size = gray.shape
result = minimize(distortion_error, initial_guess, args=(image_points, world_points, image_size), method='Powell')
optimized_params = result.x

# Extract optimized parameters
k1, k2, p1, p2, k3, fx, fy = optimized_params

# Apply the optimized distortion coefficients
camera_matrix = np.array([[fx, 0, image_size[1] / 2],
                          [0, fy, image_size[0] / 2],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Undistort the image
undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs)

# Display the original and undistorted images
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Undistorted Image')
plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
plt.show()
