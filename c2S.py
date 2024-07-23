import cv2
import numpy as np
import math

# 读取矩形图片
img_rect = cv2.imread("WechatIMG3.png")

# 定义椭圆参数
x0, y0 = img_rect.shape[1] // 2, img_rect.shape[0] // 2
a, b = img_rect.shape[1] // 2, img_rect.shape[0] // 2

#%%
# 创建空白椭圆图像
img_ellipse = np.zeros_like(img_rect)

def rect_to_ellipse(x, y, x1, y1, x2, y2, xc, yc, a, b):
    # 将矩形内的点转换为归一化坐标
    u = (x - x1) / (x2 - x1)
    v = (y - y1) / (y2 - y1)
    # 将归一化坐标转换为椭圆内的坐标
    x_ellipse = xc + a * np.cos(2 * math.pi * u)
    y_ellipse = yc + b * np.sin(2 * math.pi * v)
    return x_ellipse, y_ellipse

# 遍历椭圆图像每个像素，将其映射到矩形图像上
for x in range(img_ellipse.shape[1]):
    for y in range(img_ellipse.shape[0]):
        # 计算椭圆上对应点的坐标
        x_ellipse = x0 + int(a * ((x - 0.5 * img_ellipse.shape[1]) / (0.5 * img_ellipse.shape[1])))
        y_ellipse = y0 + int(b * ((y - 0.5 * img_ellipse.shape[0]) / (0.5 * img_ellipse.shape[0])))
        x_ellipse, y_ellipse = rect_to_ellipse(x, y, 0, 0, img_rect.shape[1], img_rect.shape[0], x0, y0, a, b)
        # 判断是否越界
        if x_ellipse < 0 or x_ellipse >= img_rect.shape[1] or y_ellipse < 0 or y_ellipse >= img_rect.shape[0]:
            continue
        # 将椭圆点的像素值换为矩形点的像素值
        # print(x,y,x_ellipse, y_ellipse)
        img_ellipse[y, x] = img_rect[int(y_ellipse), int(x_ellipse)]

# 保存椭圆图像
cv2.imwrite("ellipse.png", img_ellipse)

#%% GPT:
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def equisolid_defisheye(image, f_h, f_v):
    height, width = image.shape[:2]
    # cx, cy = round(width / 2), round(height / 2) # Center of the image
    cx, cy = width // 2, height // 2
    
    # Create an empty undistorted image
    undistorted_img = np.zeros_like(image)

    # for h in range(height):
    #     for w in range(width):
    #         # Convert pixel (i, j) to normalized coordinates
    #         x = (w - cx) / f_h #check
    #         y = (h - cy) / f_v
    #         r = np.sqrt(x**2 + y**2)
            
    #         if r > 1:
    #             continue
            
    #         # Calculate the angle theta
    #         theta = 2 * np.arcsin(r / 2)# correct
            
    #         # Calculate the corresponding pixel in the fisheye image
    #         x_fish = f_h * theta * (x / r)
    #         y_fish = f_v * theta * (y / r)
            
    #         u = int(cx + x_fish)
    #         v = int(cy + y_fish)
            
    #         if 0 <= u < width and 0 <= v < height:
    #             undistorted_img[h, w] = image[v, u]

    # return undistorted_img
    
    for h in range(height):
        for w in range(width):
            x_f = (w - cx)/f_h  #centering, in pixels, with scale
            y_f = (h - cy)/f_v
            r_f = np.sqrt(x_f**2 + y_f**2)
            
            if r_f > 1:
                continue
            
            # Calculate the angle theta
            theta = 2 * np.arcsin(r_f/ (2))#rmb to chagne back
            
            if math.isnan(theta):
                print("theta nan")
                break
            print(theta)
            # Calculate the corresponding pixel in the fisheye image
            x_p = round(f_h * np.sin(theta))
            y_p = round(f_v * np.cos(theta)) #not right
            
            u = int(cx + x_p) #shift back 
            v = int(cy + y_p)
            
            if 0 <= x_p < width and 0 <= y_p < height: #not integer index!
                undistorted_img[h, w] = image[v,u]
                

    return undistorted_img



# Load the fisheye image
fisheye_image = cv2.imread('WechatIMG3.png')

# Define the focal lengths based on FOV
focal_length_h = fisheye_image.shape[1] / (4 * np.sin(np.radians(180 / 2)))
focal_length_v = fisheye_image.shape[0] / (4 * np.sin(np.radians(85 / 2)))

# Apply the defisheye mapping
# undistorted_image = equisolid_defisheye(fisheye_image, focal_length_h, focal_length_v)

image = fisheye_image
f_h,f_v = focal_length_h, focal_length_v
height, width = image.shape[:2]
# cx, cy = round(width / 2), round(height / 2) # Center of the image
cx, cy = width // 2, height // 2

# Create an empty undistorted image
undistorted_img = np.zeros_like(image)
# list_u= []
# list_v = []
centred = np.zeros_like(image)
# list_xf = []
# list_yf = []
# list_r_f = []
# list_theta = []
for h in range(height):
    for w in range(width):
        x_f = (w - cx) #centering, in pixels, without scale
        y_f = (h - cy)
        # centred[h,w]=[x_f,y_f]
        
        r_f = np.sqrt(x_f**2 + y_f**2)
        # list_xf.append(x_f)
        # list_yf.append(y_f)
        # list_r_f.append(r_f)
        
        # if r_f > 1: #check scaling
        #     continue
        
        # # Calculate the angle theta
        theta = 2 * np.arcsin(r_f/ (2*f_h))#rmb to chagne back
        # list_theta.append(theta)
        if math.isnan(theta):
            image[h,w]= [0,0,0]
            
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                theta = 2 * np.arcsin(r_f / (2 * f_h))
                alpha = np.arctan2(y_f, x_f)
            
            # Calculate the corresponding pixel in the fisheye image
            x_p = round(f_h * np.tan(theta)*np.cos(alpha))
            y_p = round(f_h * np.tan(theta)*np.sin(alpha)) #not right
        
        u = int(cx + x_p) #shift back 
        v = int(cy + y_p)
        
        if 0 <= u < width and 0 <= v < height: #not integer index!
            undistorted_img[h, w] = image[v,u] #reposition
     
            # list_u.append(u)
            # list_v.append(v)
            
undistorted_image = undistorted_img

#%% faster version
import cv2
import numpy as np
import math

type_f= "equisolid"

# Load the fisheye image
fisheye_image = cv2.imread('WechatIMG17.png')

# Define the focal lengths based on FOV
focal_length_h = fisheye_image.shape[1] / (4 * np.sin(np.radians(180 / 4)))
focal_length_v = fisheye_image.shape[0] / (4 * np.sin(np.radians(85 / 4)))

image = fisheye_image


height, width = image.shape[:2]
cx, cy = width // 2, height // 2  # Center of the image

# Create an empty undistorted image
undistorted_img = np.zeros_like(image)

# Create meshgrid for coordinates
x_f = (np.arange(width) - cx) 
norm_xf = x_f*2/width #normalised

y_f = (np.arange(height) - cy)
norm_yf = y_f*2 /height

x_grid, y_grid = np.meshgrid(x_f, y_f)
# x_grid, y_grid = np.meshgrid(norm_xf, norm_yf) #changed here attention
r_f = np.sqrt(x_grid**2 + y_grid**2)

with np.errstate(divide='ignore', invalid='ignore'):
    if type_f == "equidistant":
        f_h = width/ (np.pi)
        f_v = height/ np.pi
        theta = r_f / f_h
    
    elif type_f == "equisolid":
        f_h = width / (4 * np.sin(np.radians(180 / 4)))
        f_v = height / (4 * np.sin(np.radians(180 / 4)))
        theta = 2 * np.arcsin(r_f / (2 * f_h))
    elif type_f == "stereographic":
        f_h = width / (4 * np.tan(np.pi/4))
        f_v = height / (4 * np.tan(np.pi/4))
        theta = 2 * np.arctan(r_f / (2 * f_h))
    else:
        print("unknown model")

    r_p = f_h * np.tan(theta) #rectilinear model
    # r_p = np.tan(r_f * np.pi/2)/(2 * np.tan(np.pi/4))
    
    
    u = cx + ((r_f/r_p)**0.3 * x_grid).astype(int)
    v = cy + ((r_f/r_p)**0.3 * y_grid).astype(int)
    
    # u = (cx + r_p * x_grid * width/2 /r_f).astype(int)
    # v = (cy + r_p * y_grid* height/2 /r_f).astype(int)
    
    # u = (cx + x_p * width/2).astype(int)
    # v = (cy + y_p * height/2).astype(int)

# Valid indices
valid_idx = (u >= 0) & (u < width) & (v >= 0) & (v < height)
undistorted_img[valid_idx] = image[v[valid_idx], u[valid_idx]]

undistorted_image = undistorted_img
# Save and display the undistorted image
cv2.imwrite('undistorted_image.jpg', undistorted_image)
plt.imshow(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))
plt.title('Undistorted Image')
plt.show()



#%%inverse version rn no difference in result :(
import cv2
import numpy as np
import math

# Load the fisheye image
fisheye_image = cv2.imread('WechatIMG3.png')

# Define the focal lengths based on FOV
focal_length_h = fisheye_image.shape[1] / (4 * np.sin(np.radians(180 / 2)))
focal_length_v = fisheye_image.shape[0] / (4 * np.sin(np.radians(85 / 2)))

image = fisheye_image
f_h, f_v = focal_length_h, focal_length_v
height, width = image.shape[:2]
cx, cy = width // 2, height // 2  # Center of the image

# Create an empty undistorted image
undistorted_img = np.zeros_like(image)

# Create meshgrid for coordinates
x_f = (np.arange(width) - cx) 
norm_xf = x_f*2/width #normalised

y_f = (np.arange(height) - cy)
norm_yf = y_f*2 /height

# x_grid, y_grid = np.meshgrid(x_f, y_f)
x_grid, y_grid = np.meshgrid(norm_xf, norm_yf) #changed here attention
r_f = np.sqrt(x_grid**2 + y_grid**2)

# Avoid division by zero
with np.errstate(divide='ignore', invalid='ignore'):
    theta = np.arctan(r_f / f_h)

    
    x_p = f_h * np.tan(theta) * np.cos(alpha) #chek alpha part
    y_p = f_h * np.tan(theta) * np.sin(alpha)
    
    r_p = 2* f_h * np.sin(theta/2)

    # u = round(cx + x_p)
    # v = round(cy + y_p)
    u = (cx + r_p * x_grid * width/2).astype(int)
    v = (cy + r_p * y_grid * height/2).astype(int)
    
    # u = (cx + x_p * width/2).astype(int)
    # v = (cy + y_p * height/2).astype(int)

# Valid indices
valid_idx = (u >= 0) & (u < width) & (v >= 0) & (v < height)
undistorted_img[valid_idx] = image[v[valid_idx], u[valid_idx]]


#%%
fig,ax=plt.subplots()

im=ax.scatter(x_grid[::10],y_grid[::10],c=scale[::10],s=1,cmap=plt.cm.jet) 

fig.colorbar(im, ax=ax, label="theta")
# plt.plot(list_xf,list_theta)
#%%
plt.imshow(cv2.cvtColor(fisheye_image, cv2.COLOR_BGR2RGB))
plt.title('Orig Fisheye Image')
#%%
