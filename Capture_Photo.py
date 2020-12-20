# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:49:06 2019

@author: Shivam
"""

# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

cap = cv2.VideoCapture(0)
cv2.waitKey(4)

ret, frame = cap.read()

while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame1 = frame.copy()
    
    # Our operations on the frame come here
    gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)
    
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the facial landmark (x, y)-coordinates to a NumPy
        # array
        shape = predictor(gray, rect)
        #print(shape)
        shape = face_utils.shape_to_np(shape)
        # loop over the (x, y)-coordinates for the facial landmarks
        # and draw them on the image
        for (x, y) in shape:
            cv2.circle(frame1, (x, y), 2, (0, 255, 0), -1)
    
    
    if cv2.waitKey(1) & 0xFF == ord('c'):
        cv2.imwrite(('Faces/Shivam.jpg'), frame)
        
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
    
    cv2.imshow("frame",frame)
    cv2.imshow("frame1",frame1)
    cv2.waitKey(1)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
