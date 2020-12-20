# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:36:25 2019

@author: Shivam
"""

import cv2
import numpy as np
img = cv2.imread("Faces/Shivam.jpg")
b,g,r = cv2.split(img)

cv2.imshow("Blue",b)
cv2.imshow("Green",g)
cv2.imshow("Red",r)

cv2.waitKey(0)
cv2.destroyAllWindows()