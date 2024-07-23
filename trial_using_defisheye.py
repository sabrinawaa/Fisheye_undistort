#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:37:46 2024

@author: sabo4ever
"""

from defisheye import Defisheye

dtype = 'linear'
format = 'fullframe'
fov = 180
pfov = 90

img = "WechatIMG3.png"
img_out = f"out/example_{dtype}_{format}_{pfov}_{fov}.jpg"

obj = Defisheye(img, dtype=dtype, format=format, fov=fov, pfov=pfov)

# To save image locally 
obj.convert(outfile=img_out)

# To use the converted image in memory

new_image = obj.convert()