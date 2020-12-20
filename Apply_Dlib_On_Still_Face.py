# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:39:59 2019

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

img = cv2.imread("Nikesh.jpg")


# Our operations on the frame come here
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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
        cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

    #Left Eyebrow

    Left_Eyebro_18_x = float(shape[17][0])
    Left_Eyebro_18_y = float(shape[17][1])

    Left_Eyebro_22_x = float(shape[21][0])
    Left_Eyebro_22_y = float(shape[21][1])

    #Right Eyebrow

    Right_Eyebro_23_x = float(shape[22][0])
    Right_Eyebro_23_y = float(shape[22][1])

    Right_Eyebro_27_x = float(shape[26][0])
    Right_Eyebro_27_y = float(shape[26][1])

    #Left Eye

    Left_Eye_37_x = float(shape[36][0])
    Left_Eye_37_y = float(shape[36][1])

    Left_Eye_40_x = float(shape[39][0])
    Left_Eye_40_y = float(shape[39][1])


    #Right Eye

    Right_Eye_43_x = float(shape[42][0])
    Right_Eye_43_y = float(shape[42][1])

    Right_Eye_46_x = float(shape[45][0])
    Right_Eye_46_y = float(shape[45][1])

    #Nose

    Nose_28_x = float(shape[27][0])
    Nose_28_y = float(shape[27][1])

    Nose_31_x = float(shape[30][0])
    Nose_31_y = float(shape[30][1])

    # Nose_Lower

    Nose_Lower_32_x = float(shape[31][0])
    Nose_Lower_32_y = float(shape[31][1])

    Nose_Lower_36_x = float(shape[35][0])
    Nose_Lower_36_y = float(shape[35][1])

    #Nose_Tip

    Nose_Tip_31_x = float(shape[32][0])
    Nose_Tip_31_y = float(shape[32][1])

    #Chin

    Chin_9_x = float(shape[8][0])
    Chin_9_y = float(shape[8][1])

    #Lips

    Lips_49_x = float(shape[48][0])
    Lips_49_y = float(shape[48][1])

    Lips_55_x = float(shape[54][0])
    Lips_55_y = float(shape[54][1])

    #Face Horizontal

    Face_Horizontal_1_x = float(shape[0][0])
    Face_Horizontal_1_y = float(shape[0][1])

    Face_Horizontal_17_x = float(shape[16][0])
    Face_Horizontal_17_y = float(shape[16][1])

    #Distance

    Distance_Left_Eyebro = math.hypot((Left_Eyebro_18_x-Left_Eyebro_22_x),(Left_Eyebro_18_y-Left_Eyebro_22_y))
    Distance_Right_Eyebro = math.hypot((Right_Eyebro_23_x - Right_Eyebro_27_x), (Right_Eyebro_23_y - Right_Eyebro_27_y))
    Distance_Left_Eye = math.hypot((Left_Eye_37_x - Left_Eye_40_x),(Left_Eye_37_y - Left_Eye_40_y))
    Distance_Right_Eye = math.hypot((Right_Eye_43_x - Right_Eye_46_x),(Right_Eye_43_y - Right_Eye_46_y))
    Distance_Nose = math.hypot((Nose_28_x - Nose_31_x), (Nose_28_y - Nose_31_y))
    Distance_Nose_Lower = math.hypot((Nose_Lower_32_x - Nose_Lower_36_x), (Nose_Lower_32_y - Nose_Lower_36_y))
    Distance_Lips = math.hypot((Lips_49_x - Lips_55_x), (Lips_49_y - Lips_55_y))
    Distance_NoseTip_Chin = math.hypot((Nose_Tip_31_x - Chin_9_x), (Nose_Tip_31_y - Chin_9_y))
    Distance_Face_Horizontal = math.hypot((Face_Horizontal_1_x - Face_Horizontal_17_x), (Face_Horizontal_1_y - Face_Horizontal_17_y))

    #Some important calculation
    Eye_Length = (Distance_Left_Eye + Distance_Left_Eye)/2


    #Print Distance

    #print("Distance_Left_Eyebro",Distance_Left_Eyebro)        #Eyebrow Data is not reliable
    #print("Distance_Right_Eyebro",Distance_Right_Eyebro)      #Eyebrow Date is not reliable
    #print("Distance_Right_Eye", Distance_Right_Eye)
    #print("Distance_Left_Eye", Distance_Left_Eye)
    #print("Distance_Nose", Distance_Nose)
    #print("Distance_Nose_Lower", Distance_Nose_Lower)
    #print("Distance_Lips", Distance_Lips)

    #Calculating Ratios
    Ratio_LeftEye_RightEye = Distance_Left_Eye/Distance_Right_Eye
    Ratio_Nose_NoseLower = Distance_Nose/Distance_Nose_Lower
    Ratio_Eye_Nose = Eye_Length/Distance_Nose
    Ratio_Eye_NoseLower = Eye_Length/Distance_Nose_Lower
    Ratio_NoseTipChin_Eye = Distance_NoseTip_Chin/Eye_Length
    Ratio_Lip_Eye = Distance_Lips/Eye_Length
    Ratio_HorizontalFace_Eye = Distance_Face_Horizontal/Eye_Length

    #Print Ratio
    print("Ratio_LeftEye_RightEye = ",Ratio_LeftEye_RightEye)
    print("Ratio_Nose_NoseLower =",Ratio_Nose_NoseLower)
    print("Ratio_Eye_Nose =",Ratio_Eye_Nose)
    print("Ratio_Eye_NoseLower =",Ratio_Eye_NoseLower)
    print("Ratio_NoseTipChin_Eye =",Ratio_NoseTipChin_Eye)
    print("Ratio_Lip_Eye =",Ratio_Lip_Eye)
    print("Ratio_HorizontalFace_Eye =",Ratio_HorizontalFace_Eye)

cv2.imshow("Dlib_Image",img)
cv2.waitKey(5000)

cv2.destroyAllWindows()