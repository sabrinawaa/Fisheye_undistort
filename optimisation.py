import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load the image
# image_path = '../WechatIMG17.png'
num = str(127)
image_path = 'chessboard/bkg-' + num+ '.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

#%%

# Placeholder grid points (these should be your manually marked points)
# Example: image_points should be in pixel coordinates
image_points = np.load("gridpoints/corners"+ num +".npy").astype(np.float32)

# Corresponding ideal grid points in undistorted space
# These points form a regular grid
if num == str(115): #flat， also 118,119,120
    top_left = np.array([780, 528])
    top_right = np.array([700, 100])
    bottom_right = np.array([1168, 518])
    bottom_left = np.array([50, 500])
    
    x = np.linspace(780,1168,11)
    y = np.linspace(318,518,8)
    
elif num == str(116):
    x = np.linspace(318,733,11)
    y = np.linspace(255,516,8)
    
# elif num == str(120):
#     x = np.linspace(785,1225,11)
#     y = np.linspace(393,609,8)     
    
elif num == str(121):
    x = np.linspace(660,1208,11)
    y = np.linspace(241,540,8)   
    
elif num == str(122):
    x = np.linspace(904,1227,11)
    y = np.linspace(61,343,8)     
    
elif num == str(124):
    x = np.linspace(41,52,11)
    y = np.linspace(243,587,8)  
    
elif num == str(125):
    x = np.linspace(1127,1555,11)
    y = np.linspace(277,604,8)  
    
elif num == str(126):
    x = np.linspace(730,1235,11)
    y = np.linspace(317,637,8)  
    
elif num == str(127):
    x = np.linspace(60,1578,21)
    y = np.linspace(57,671,10)  
    
else:
    print("num not founnd")

world_points = np.meshgrid(x,y)
world_points = np.vstack([world_points[0].ravel(), world_points[1].ravel()]).T.reshape(-1,1,2)
for i in world_points:
    x, y = i.ravel().astype(int)
    cv2.circle(image, (x,y),3,255,-1)
plt.imshow(image)


#%%
# Initial guess for distortion coefficients (k1, k2, p1, p2, k3) and focal lengths (fx, fy)
initial_guess = np.array([-0.0, 0.7, 0.0, -0.0, 0.0, 0.0, 0.0,0.0, image.shape[1]//2, image.shape[0]//2])

# Define the optimization function
def distortion_error(params, image_points, world_points, image_size):
    k1, k2, p1, p2, k3,k4,k5,k6, fx, fy = params
    camera_matrix = np.array([[fx, 0, image_size[1]/ 2],
                              [0, fy, image_size[0] / 2],
                              [0, 0, 1]], dtype=np.float32)
    dist_coeffs = np.array([k1, k2, p1, p2, k3, k4, k5, k6], dtype=np.float32)
    
    undistorted_points = cv2.undistortPoints(image_points, camera_matrix, dist_coeffs, P=camera_matrix)
    
    # Calculate the error as the difference between undistorted points and world points
    error = np.linalg.norm(undistorted_points - world_points, axis=1)
    return np.sum(error)

# Perform the optimization
image_size = gray.shape
result = minimize(distortion_error, initial_guess, args=(image_points, world_points, image_size), method='Powell')
optimized_params = result.x

# Extract optimized parameters
k1, k2, p1, p2, k3, k4, k5, k6, fx, fy = optimized_params

# Apply the optimized distortion coefficients
mtx = np.array([[fx, 0, image_size[1] / 2],
                [0, fy, image_size[0] / 2],
                [0, 0, 1]], dtype=np.float32)
dist = np.array([k1, k2, p1, p2, k3, k4,k5,k6], dtype=np.float32)

# Undistort the image
undistorted_image = cv2.undistort(image, mtx, dist)


# Display the original and undistorted images
plt.figure(figsize=(15, 7))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.subplot(1, 2, 2)
plt.title('Undistorted Image')
plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
plt.show()

#%%
image2 = cv2.imread("../WechatIMG17.png")
undistorted_image2 = cv2.undistort(image2, mtx, dist)
plt.imshow(cv2.cvtColor(undistorted_image2, cv2.COLOR_BGR2RGB))
plt.savefig("optimisation_undistorted.jpg")
