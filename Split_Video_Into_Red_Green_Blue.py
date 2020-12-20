# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 10:51:01 2019

@author: Shivam
"""

# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math

cap = cv2.VideoCapture(0)
cv2.waitKey(4)

ret, frame = cap.read()

while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame1 = frame.copy()
    
    b,g,r = cv2.split(frame1)
    
    cv2.imshow("Blue",b)
    cv2.imshow("Green",g)
    cv2.imshow("Red",r)

    
#    cv2.imshow("frame",frame)
#    cv2.imshow("frame1",frame1)
    cv2.waitKey(1)
    
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
