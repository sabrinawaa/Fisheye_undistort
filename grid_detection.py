#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 11:58:03 2024

@author: sabo4ever
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 11:34:19 2024

@author: sabo4ever
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt

#ccreate binary image
#%%
def increase_sharpness(image_path, threshold_value=127):
    # Read the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Check if the image is loaded properly
    if img is None:
        raise ValueError("Image not found or unable to load the image.")
    
    # Apply thresholding to convert the image to binary
    _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)
    
    return binary_img

for i in [118,119,120,121,122,123]:
    num = str(i)
    img = 'chessboard/' +num +'.png'
    
    threshold_value = 127  # You can adjust this value as needed
    
    binary_img = increase_sharpness(img, threshold_value)
    
    # Display the result
    # plt.imshow(binary_img, cmap='gray')
    # plt.title('Binary Image')
    # plt.axis('off')
    # plt.show()
    # Save the result
    cv2.imwrite('chessboard/binary_'+num+'.jpg', binary_img)
#%%

import cv2
import numpy as np
from matplotlib import pyplot as plt

num = str(118)
img = cv2.imread('chessboard/white_bkgd/' +num +'.png')
new = img.copy()
#invert 
imagem = cv2.bitwise_not(img)
gray = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 90, 0.01, 50, blockSize = 5)
corners = np.intp(corners)
for i in corners:
    # import pdb; pdb.set_trace()
    x, y = i.ravel()
    cv2.circle(imagem, (x,y),10,255,-1)
plt.imshow(imagem)
# cv2.imwrite('hough_img.png',imagem)

#%%
cv2.drawChessboardCorners(imagem, (11, 8), corners, True)
#%%
xx,yy = 665,538
thres = 3


for i in range(len(corners)):
    [[x,y]] = corners[i]
    if abs(xx-x)<thres and abs(yy-y)<thres:
        print(x,y)
        # cv2.circle(imagem, (x,y),5,(0,255,0),-1)
        # plt.imshow(imagem)
        idx = i
 #%% 

corners1 = np.delete(corners,idx,axis=0) #change dynamically
corners1 = corners1.reshape(-1, 1, 2)      
#%%
corners1 = corners
if len(corners1) == 88:
    print("file saved")
    np.save(num,corners1)


 

