# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 19:05:31 2019

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

#img = np.array([[[0]]])

def Ratios_Of_Image(img):

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
        #for (x, y) in shape:
        #    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        #Left Face Curve
        
        L2_x = float(shape[1][0])
        L2_y = float(shape[1][1])
        
        L3_x = float(shape[2][0])
        L3_y = float(shape[2][1])
        
        L4_x = float(shape[3][0])
        L4_y = float(shape[3][1])
        
        L5_x = float(shape[4][0])
        L5_y = float(shape[4][1])
        
        L6_x = float(shape[5][0])
        L6_y = float(shape[5][1])
        
        L7_x = float(shape[6][0])
        L7_y = float(shape[6][1])
        
        L8_x = float(shape[7][0])
        L8_y = float(shape[7][1])
        
        #Right Face Curve
        
        R10_x = float(shape[9][0])
        R10_y = float(shape[9][1])
        
        R11_x = float(shape[10][0])
        R11_y = float(shape[10][1])
        
        R12_x = float(shape[11][0])
        R12_y = float(shape[11][1])
        
        R13_x = float(shape[12][0])
        R13_y = float(shape[12][1])
        
        R14_x = float(shape[13][0])
        R14_y = float(shape[13][1])
        
        R15_x = float(shape[14][0])
        R15_y = float(shape[14][1])
        
        R16_x = float(shape[15][0])
        R16_y = float(shape[15][1])
        
    
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
        Distance_Left_Right_Eyebro = math.hypot((Left_Eyebro_18_x-Right_Eyebro_27_x),(Left_Eyebro_18_y-Right_Eyebro_27_y))
        Distance_Left_Eye = math.hypot((Left_Eye_37_x - Left_Eye_40_x),(Left_Eye_37_y - Left_Eye_40_y))
        Distance_Right_Eye = math.hypot((Right_Eye_43_x - Right_Eye_46_x),(Right_Eye_43_y - Right_Eye_46_y))
        Distance_Nose = math.hypot((Nose_28_x - Nose_31_x), (Nose_28_y - Nose_31_y))
        Distance_Nose_Lower = math.hypot((Nose_Lower_32_x - Nose_Lower_36_x), (Nose_Lower_32_y - Nose_Lower_36_y))
        Distance_Lips = math.hypot((Lips_49_x - Lips_55_x), (Lips_49_y - Lips_55_y))
        Distance_NoseTip_Chin = math.hypot((Nose_Tip_31_x - Chin_9_x), (Nose_Tip_31_y - Chin_9_y))
        Distance_Face_Horizontal = math.hypot((Face_Horizontal_1_x - Face_Horizontal_17_x), (Face_Horizontal_1_y - Face_Horizontal_17_y))
    
        
        #Left Face Curve Distance With Right Eye_43
        
        Distance_2_43 = math.hypot((L2_x-Right_Eye_43_x),(L2_y-Right_Eye_43_y))
        Distance_3_43 = math.hypot((L3_x-Right_Eye_43_x),(L3_y-Right_Eye_43_y))
        Distance_4_43 = math.hypot((L4_x-Right_Eye_43_x),(L4_y-Right_Eye_43_y))
        Distance_5_43 = math.hypot((L5_x-Right_Eye_43_x),(L5_y-Right_Eye_43_y))
        Distance_6_43 = math.hypot((L6_x-Right_Eye_43_x),(L6_y-Right_Eye_43_y))
        Distance_7_43 = math.hypot((L7_x-Right_Eye_43_x),(L7_y-Right_Eye_43_y))
        Distance_8_43 = math.hypot((L8_x-Right_Eye_43_x),(L8_y-Right_Eye_43_y))
        
        
        #Right Face Curve Distance With Left Eye_40
        
        Distance_10_40 = math.hypot((R10_x-Left_Eye_40_x),(R10_y-Left_Eye_40_y))
        Distance_11_40 = math.hypot((R11_x-Left_Eye_40_x),(R11_y-Left_Eye_40_y))
        Distance_12_40 = math.hypot((R12_x-Left_Eye_40_x),(R12_y-Left_Eye_40_y))
        Distance_13_40 = math.hypot((R13_x-Left_Eye_40_x),(R13_y-Left_Eye_40_y))
        Distance_14_40 = math.hypot((R14_x-Left_Eye_40_x),(R14_y-Left_Eye_40_y))
        Distance_15_40 = math.hypot((R15_x-Left_Eye_40_x),(R15_y-Left_Eye_40_y))
        Distance_16_40 = math.hypot((R16_x-Left_Eye_40_x),(R16_y-Left_Eye_40_y))
        
        #Face Horizontal Distance
        
        Distance_2_16 = math.hypot((L2_x-R16_x),(L2_y-R16_y))
        Distance_3_15 = math.hypot((L3_x-R15_x),(L3_y-R15_y))
        Distance_4_14 = math.hypot((L4_x-R14_x),(L4_y-R14_y))
        Distance_5_13 = math.hypot((L5_x-R13_x),(L5_y-R13_y))
        Distance_6_12 = math.hypot((L6_x-R12_x),(L6_y-R12_y))
        Distance_7_11 = math.hypot((L7_x-R11_x),(L7_y-R11_y))
        Distance_8_10 = math.hypot((L8_x-R10_x),(L8_y-R10_y))
        
        
        #Some important calculation
        Eye_Length = (Distance_Left_Eye + Distance_Left_Eye)/2
    
    
        #Print Distance
    
        #print("Distance_Left_Eyebro",Distance_Left_Eyebro)        #Eyebrow Data is not reliable
        #print("Distance_Right_Eyebro",Distance_Right_Eyebro)      #Eyebrow Date is not reliable
        #print(Distance_Left_Right_Eyebro)
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
        Ratio_LeftRightEyebro_Eye = Distance_Left_Right_Eyebro/Eye_Length
        
        #Face Curve Ratio
        
        Ratio_2_43_Eye = Distance_2_43/Eye_Length
        Ratio_3_43_Eye = Distance_3_43/Eye_Length
        Ratio_4_43_Eye = Distance_4_43/Eye_Length
        Ratio_5_43_Eye = Distance_5_43/Eye_Length
        Ratio_6_43_Eye = Distance_6_43/Eye_Length
        Ratio_7_43_Eye = Distance_7_43/Eye_Length
        Ratio_8_43_Eye = Distance_8_43/Eye_Length
        
        Ratio_10_40_Eye = Distance_10_40/Eye_Length
        Ratio_11_40_Eye = Distance_11_40/Eye_Length
        Ratio_12_40_Eye = Distance_12_40/Eye_Length
        Ratio_13_40_Eye = Distance_13_40/Eye_Length
        Ratio_14_40_Eye = Distance_14_40/Eye_Length
        Ratio_15_40_Eye = Distance_15_40/Eye_Length
        Ratio_16_40_Eye = Distance_16_40/Eye_Length
        
        #Face Horizontal Ratio
        
        Ratio_2_16_Eye = Distance_2_16/Eye_Length
        Ratio_3_15_Eye = Distance_3_15/Eye_Length
        Ratio_4_14_Eye = Distance_4_14/Eye_Length
        Ratio_5_13_Eye = Distance_5_13/Eye_Length
        Ratio_6_12_Eye = Distance_6_12/Eye_Length
        Ratio_7_11_Eye = Distance_7_11/Eye_Length
        Ratio_8_10_Eye = Distance_8_10/Eye_Length
        
        
        
    return Ratio_LeftEye_RightEye,Ratio_Nose_NoseLower,Ratio_Eye_Nose,Ratio_Eye_NoseLower,Ratio_NoseTipChin_Eye,Ratio_Lip_Eye,Ratio_HorizontalFace_Eye,Ratio_LeftRightEyebro_Eye,Ratio_2_43_Eye,Ratio_3_43_Eye,Ratio_4_43_Eye,Ratio_5_43_Eye,Ratio_6_43_Eye,Ratio_7_43_Eye,Ratio_8_43_Eye,Ratio_10_40_Eye,Ratio_11_40_Eye,Ratio_12_40_Eye,Ratio_13_40_Eye,Ratio_14_40_Eye,Ratio_15_40_Eye,Ratio_16_40_Eye,Ratio_2_16_Eye,Ratio_3_15_Eye,Ratio_4_14_Eye,Ratio_5_13_Eye,Ratio_6_12_Eye,Ratio_7_11_Eye,Ratio_8_10_Eye