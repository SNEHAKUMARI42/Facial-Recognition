# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 17:27:55 2019

@author: Shivam
"""

# import the necessary packages
from imutils import face_utils
import dlib
import cv2
import numpy as np
import math
import Error_Window
import Read_Image_Data
import os

Upper_Limit, Lower_Limit = Error_Window.Error_Allowed(10,10)

#Percentage of error allowed
Allowed_Error = 2
Allowed_Error_Higher = 15 
Allowed_Error_Lower = 8

Allowed_Error_Ratio_Nose_NoseLower = 7
Allowed_Error_Ratio_Eye_Nose = 7
Allowed_Error_Ratio_Eye_NoseLower = 7
Allowed_Error_Ratio_NoseTipChin_Eye = 7
Allowed_Error_Ratio_Lip_Eye = 8
Allowed_Error_Ratio_HorizontalFace_Eye = 8
Allowed_Error_Ratio_LeftRightEyebro_Eye = 8

Allowed_Error_Ratio_2_43_Eye = 6
Allowed_Error_Ratio_3_43_Eye = 6
Allowed_Error_Ratio_4_43_Eye = 6
Allowed_Error_Ratio_5_43_Eye = 6
Allowed_Error_Ratio_6_43_Eye = 6
Allowed_Error_Ratio_7_43_Eye = 6
Allowed_Error_Ratio_8_43_Eye = 6

Allowed_Error_Ratio_10_40_Eye = 6
Allowed_Error_Ratio_11_40_Eye = 6
Allowed_Error_Ratio_12_40_Eye = 6
Allowed_Error_Ratio_13_40_Eye = 6
Allowed_Error_Ratio_14_40_Eye = 6
Allowed_Error_Ratio_15_40_Eye = 6
Allowed_Error_Ratio_16_40_Eye = 6

Allowed_Error_Ratio_2_16_Eye = 6
Allowed_Error_Ratio_3_15_Eye = 6
Allowed_Error_Ratio_4_14_Eye = 6
Allowed_Error_Ratio_5_13_Eye = 6
Allowed_Error_Ratio_6_12_Eye = 6
Allowed_Error_Ratio_7_11_Eye = 6
Allowed_Error_Ratio_8_10_Eye = 6

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

def load_images_from_folder(folder):
    images = []
    filenames  = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            filenames.append(filename)
    return images,filenames

images,filenames = load_images_from_folder('Faces')

#print(len(images))

Number_Of_Images = len(images)


Img_Ratio_LeftEye_RightEye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_Nose_NoseLower = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_Eye_Nose = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_Eye_NoseLower = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_NoseTipChin_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_Lip_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_HorizontalFace_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_LeftRightEyebro_Eye = np.empty(Number_Of_Images, dtype=float)

Img_Ratio_2_43_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_3_43_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_4_43_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_5_43_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_6_43_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_7_43_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_8_43_Eye = np.empty(Number_Of_Images, dtype=float)

Img_Ratio_10_40_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_11_40_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_12_40_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_13_40_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_14_40_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_15_40_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_16_40_Eye = np.empty(Number_Of_Images, dtype=float)

Img_Ratio_2_16_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_3_15_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_4_14_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_5_13_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_6_12_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_7_11_Eye = np.empty(Number_Of_Images, dtype=float)
Img_Ratio_8_10_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__LeftEye_RightEye = np.empty(Number_Of_Images, dtype=float)
Lower__LeftEye_RightEye = np.empty(Number_Of_Images, dtype=float)

Upper__Nose_NoseLower = np.empty(Number_Of_Images, dtype=float)
Lower__Nose_NoseLower = np.empty(Number_Of_Images, dtype=float)

Upper__Eye_Nose = np.empty(Number_Of_Images, dtype=float)
Lower__Eye_Nose = np.empty(Number_Of_Images, dtype=float)

Upper__Eye_NoseLower = np.empty(Number_Of_Images, dtype=float)
Lower__Eye_NoseLower = np.empty(Number_Of_Images, dtype=float)

Upper__NoseTipChin_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__NoseTipChin_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__Lip_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__Lip_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__HorizontalFace_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__HorizontalFace_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__LeftRightEyebro_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__LeftRightEyebro_Eye = np.empty(Number_Of_Images, dtype=float)


Upper__2_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__2_43_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__3_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__3_43_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__4_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__4_43_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__5_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__5_43_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__6_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__6_43_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__7_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__7_43_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__8_43_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__8_43_Eye = np.empty(Number_Of_Images, dtype=float)


Upper__10_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__10_40_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__11_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__11_40_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__12_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__12_40_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__13_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__13_40_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__14_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__14_40_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__15_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__15_40_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__16_40_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__16_40_Eye = np.empty(Number_Of_Images, dtype=float)


Upper__2_16_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__2_16_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__3_15_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__3_15_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__4_14_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__4_14_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__5_13_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__5_13_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__6_12_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__6_12_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__7_11_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__7_11_Eye = np.empty(Number_Of_Images, dtype=float)

Upper__8_10_Eye = np.empty(Number_Of_Images, dtype=float)
Lower__8_10_Eye = np.empty(Number_Of_Images, dtype=float)



'''Creating an boolean array with "False" value as default'''
is_person = np.zeros(Number_Of_Images, dtype=bool)
count = np.zeros(Number_Of_Images, dtype=int)


for i in range(0,len(images)):
    
    #Taking Image(Image of person whose face is to be detected) Ratios
    Img_Ratio_LeftEye_RightEye[i], Img_Ratio_Nose_NoseLower[i], Img_Ratio_Eye_Nose[i], Img_Ratio_Eye_NoseLower[i], Img_Ratio_NoseTipChin_Eye[i], Img_Ratio_Lip_Eye[i], Img_Ratio_HorizontalFace_Eye[i] ,Img_Ratio_LeftRightEyebro_Eye[i] ,Img_Ratio_2_43_Eye[i] ,Img_Ratio_3_43_Eye[i] ,Img_Ratio_4_43_Eye[i] ,Img_Ratio_5_43_Eye[i] ,Img_Ratio_6_43_Eye[i] ,Img_Ratio_7_43_Eye[i] ,Img_Ratio_8_43_Eye[i] ,Img_Ratio_10_40_Eye[i] ,Img_Ratio_11_40_Eye[i] ,Img_Ratio_12_40_Eye[i] ,Img_Ratio_13_40_Eye[i] ,Img_Ratio_14_40_Eye[i] ,Img_Ratio_15_40_Eye[i] ,Img_Ratio_16_40_Eye[i] ,Img_Ratio_2_16_Eye[i] ,Img_Ratio_3_15_Eye[i] ,Img_Ratio_4_14_Eye[i] ,Img_Ratio_5_13_Eye[i] ,Img_Ratio_6_12_Eye[i] ,Img_Ratio_7_11_Eye[i] ,Img_Ratio_8_10_Eye[i] =  Read_Image_Data.Ratios_Of_Image(images[i])
    
    #Taking Upper and Lower Limit of various Ratios
    
    
    #LeftEye_RightEye
    Upper__LeftEye_RightEye[i],Lower__LeftEye_RightEye[i] = Error_Window.Error_Allowed(Img_Ratio_LeftEye_RightEye[i],Allowed_Error)
    
    #Nose_NoseLower
    Upper__Nose_NoseLower[i],Lower__Nose_NoseLower[i] = Error_Window.Error_Allowed(Img_Ratio_Nose_NoseLower[i],Allowed_Error_Ratio_Nose_NoseLower)
    
    #Eye_Nose
    Upper__Eye_Nose[i],Lower__Eye_Nose[i] = Error_Window.Error_Allowed(Img_Ratio_Eye_Nose[i],Allowed_Error_Ratio_Eye_Nose)
    
    #Eye_NoseLower
    Upper__Eye_NoseLower[i],Lower__Eye_NoseLower[i] = Error_Window.Error_Allowed(Img_Ratio_Eye_NoseLower[i],Allowed_Error_Ratio_Eye_NoseLower)
    
    #NoseTipChin_Eye
    Upper__NoseTipChin_Eye[i],Lower__NoseTipChin_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_NoseTipChin_Eye[i],Allowed_Error_Ratio_NoseTipChin_Eye)
    
    #Lip_Eye
    Upper__Lip_Eye[i],Lower__Lip_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_Lip_Eye[i],Allowed_Error_Ratio_Lip_Eye)
    
    #HorizontalFace_Eye
    Upper__HorizontalFace_Eye[i],Lower__HorizontalFace_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_HorizontalFace_Eye[i],Allowed_Error_Ratio_HorizontalFace_Eye)
    
    
    #LeftRightEyebro_Eye
    Upper__LeftRightEyebro_Eye[i],Lower__LeftRightEyebro_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_LeftRightEyebro_Eye[i],Allowed_Error_Ratio_LeftRightEyebro_Eye)
    
    
    #Left Curve
    Upper__2_43_Eye[i],Lower__2_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_2_43_Eye[i],Allowed_Error_Ratio_2_43_Eye)
    Upper__3_43_Eye[i],Lower__3_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_3_43_Eye[i],Allowed_Error_Ratio_3_43_Eye)
    Upper__4_43_Eye[i],Lower__4_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_4_43_Eye[i],Allowed_Error_Ratio_4_43_Eye)
    Upper__5_43_Eye[i],Lower__5_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_5_43_Eye[i],Allowed_Error_Ratio_5_43_Eye)
    Upper__6_43_Eye[i],Lower__6_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_6_43_Eye[i],Allowed_Error_Ratio_6_43_Eye)
    Upper__7_43_Eye[i],Lower__7_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_7_43_Eye[i],Allowed_Error_Ratio_7_43_Eye)
    Upper__8_43_Eye[i],Lower__8_43_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_8_43_Eye[i],Allowed_Error_Ratio_8_43_Eye)
    
    #Right Curve
    Upper__10_40_Eye[i],Lower__10_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_10_40_Eye[i],Allowed_Error_Ratio_10_40_Eye)
    Upper__11_40_Eye[i],Lower__11_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_11_40_Eye[i],Allowed_Error_Ratio_11_40_Eye)
    Upper__12_40_Eye[i],Lower__12_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_12_40_Eye[i],Allowed_Error_Ratio_12_40_Eye)
    Upper__13_40_Eye[i],Lower__13_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_13_40_Eye[i],Allowed_Error_Ratio_13_40_Eye)
    Upper__14_40_Eye[i],Lower__14_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_14_40_Eye[i],Allowed_Error_Ratio_14_40_Eye)
    Upper__15_40_Eye[i],Lower__15_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_15_40_Eye[i],Allowed_Error_Ratio_15_40_Eye)
    Upper__16_40_Eye[i],Lower__16_40_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_16_40_Eye[i],Allowed_Error_Ratio_16_40_Eye)
    
    #Horizontal Face
    Upper__2_16_Eye[i],Lower__2_16_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_2_16_Eye[i],Allowed_Error_Ratio_2_16_Eye)
    Upper__3_15_Eye[i],Lower__3_15_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_3_15_Eye[i],Allowed_Error_Ratio_3_15_Eye)
    Upper__4_14_Eye[i],Lower__4_14_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_4_14_Eye[i],Allowed_Error_Ratio_4_14_Eye)
    Upper__5_13_Eye[i],Lower__5_13_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_5_13_Eye[i],Allowed_Error_Ratio_5_13_Eye)
    Upper__6_12_Eye[i],Lower__6_12_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_6_12_Eye[i],Allowed_Error_Ratio_6_12_Eye)
    Upper__7_11_Eye[i],Lower__7_11_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_7_11_Eye[i],Allowed_Error_Ratio_7_11_Eye)
    Upper__8_10_Eye[i],Lower__8_10_Eye[i] = Error_Window.Error_Allowed(Img_Ratio_8_10_Eye[i],Allowed_Error_Ratio_8_10_Eye)



cap = cv2.VideoCapture(0)
print(cap.isOpened())


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)



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
        


        #Print Ratio
        #print("Ratio_LeftEye_RightEye = ",Ratio_LeftEye_RightEye)
        #print("Ratio_Nose_NoseLower =",Ratio_Nose_NoseLower)
        #print("Ratio_Eye_Nose =",Ratio_Eye_Nose)
        #print("Ratio_Eye_NoseLower =",Ratio_Eye_NoseLower)
        #print("Ratio_NoseTipChin_Eye =",Ratio_NoseTipChin_Eye)
        #print("Ratio_Lip_Eye =",Ratio_Lip_Eye)
        #print("Ratio_HorizontalFace_Eye =",Ratio_HorizontalFace_Eye)

        for i in range(0,len(images)):
    
            #person Face ID
            if(Ratio_Nose_NoseLower <= Upper__Nose_NoseLower[i] and Ratio_Nose_NoseLower >= Lower__Nose_NoseLower[i]):
                First_ID = True
            else:
                First_ID = False
    
            if(Ratio_LeftEye_RightEye <= Upper__LeftEye_RightEye[i] and Ratio_LeftEye_RightEye >= Lower__LeftEye_RightEye[i]):
                Second_ID = True
            else:
                Second_ID = False
    
            if(Ratio_Eye_Nose <= Upper__Eye_Nose[i] and Ratio_Eye_Nose >= Lower__Eye_Nose[i]):
                Third_ID = True
            else:
                Third_ID = False
    
            if(Ratio_NoseTipChin_Eye <= Upper__NoseTipChin_Eye[i]  and  Ratio_NoseTipChin_Eye >= Lower__NoseTipChin_Eye[i] ):
                Fourth_ID = True
            else:
                Fourth_ID = False
    
            if(Ratio_Lip_Eye <= Upper__Lip_Eye[i]  and  Ratio_Lip_Eye >= Lower__Lip_Eye[i] ):
                Fifth_ID = True
            else:
                Fifth_ID = False
    
            if(Ratio_HorizontalFace_Eye <= Upper__HorizontalFace_Eye[i]  and  Ratio_HorizontalFace_Eye >= Lower__HorizontalFace_Eye[i] ):
                Sixth_ID = True
            else:
                Sixth_ID = False
    
            if(Ratio_LeftRightEyebro_Eye <= Upper__LeftRightEyebro_Eye[i]  and  Ratio_LeftRightEyebro_Eye >= Lower__LeftRightEyebro_Eye[i] ):
                Seventh_ID = True
                
            else:
                Seventh_ID = False
    

            if(Ratio_LeftEye_RightEye <= Upper__LeftEye_RightEye[i]  and  Ratio_LeftEye_RightEye >= Lower__LeftEye_RightEye[i] ):
                Eighth_ID = True
                
            else:
                Eighth_ID = False


            #Left Curve ID


            if(Ratio_2_43_Eye <= Upper__2_43_Eye[i]  and  Ratio_2_43_Eye >= Lower__2_43_Eye[i] ):
                L2_ID = True
                
            else:
                L2_ID = False
                
            
            if(Ratio_3_43_Eye <= Upper__3_43_Eye[i]  and  Ratio_3_43_Eye >= Lower__3_43_Eye[i] ):
                L3_ID = True
                
            else:
                L3_ID = False
                
            
            if(Ratio_4_43_Eye <= Upper__4_43_Eye[i]  and  Ratio_4_43_Eye >= Lower__4_43_Eye[i] ):
                L4_ID = True
                
            else:
                L4_ID = False
                
            
            if(Ratio_5_43_Eye <= Upper__5_43_Eye[i]  and  Ratio_5_43_Eye >= Lower__5_43_Eye[i] ):
                L5_ID = True
                
            else:
                L5_ID = False
                
            
            if(Ratio_6_43_Eye <= Upper__6_43_Eye[i]  and  Ratio_6_43_Eye >= Lower__6_43_Eye[i] ):
                L6_ID = True
                
            else:
                L6_ID = False
                
            
            if(Ratio_7_43_Eye <= Upper__7_43_Eye[i]  and  Ratio_7_43_Eye >= Lower__7_43_Eye[i] ):
                L7_ID = True
                
            else:
                L7_ID = False
                
            
            if(Ratio_8_43_Eye <= Upper__8_43_Eye[i]  and  Ratio_8_43_Eye >= Lower__8_43_Eye[i] ):
                L8_ID = True
                
            else:
                L8_ID = False


            #Right Curve ID
            
            
            if(Ratio_10_40_Eye <= Upper__10_40_Eye[i]  and  Ratio_10_40_Eye >= Lower__10_40_Eye[i] ):
                R10_ID = True
                
            else:
                R10_ID = False
                
            
            if(Ratio_11_40_Eye <= Upper__11_40_Eye[i]  and  Ratio_11_40_Eye >= Lower__11_40_Eye[i] ):
                R11_ID = True
                
            else:
                R11_ID = False
                
            
            if(Ratio_12_40_Eye <= Upper__12_40_Eye[i]  and  Ratio_12_40_Eye >= Lower__12_40_Eye[i] ):
                R12_ID = True
                
            else:
                R12_ID = False
                
            
            if(Ratio_13_40_Eye <= Upper__13_40_Eye[i]  and  Ratio_13_40_Eye >= Lower__13_40_Eye[i] ):
                R13_ID = True
                
            else:
                R13_ID = False
                
            
            if(Ratio_14_40_Eye <= Upper__14_40_Eye[i]  and  Ratio_14_40_Eye >= Lower__14_40_Eye[i] ):
                R14_ID = True
                
            else:
                R14_ID = False
                
            
            if(Ratio_15_40_Eye <= Upper__15_40_Eye[i]  and  Ratio_15_40_Eye >= Lower__15_40_Eye[i] ):
                R15_ID = True
                
            else:
                R15_ID = False
                
            
            if(Ratio_16_40_Eye <= Upper__16_40_Eye[i]  and  Ratio_16_40_Eye >= Lower__16_40_Eye[i] ):
                R16_ID = True
                
            else:
                R16_ID = False
                
            
            #Horizontal Face ID



            if(Ratio_2_16_Eye <= Upper__2_16_Eye[i]  and  Ratio_2_16_Eye >= Lower__2_16_Eye[i] ):
                H2_16_ID = True
                
            else:
                H2_16_ID = False
                
            
            if(Ratio_3_15_Eye <= Upper__3_15_Eye[i]  and  Ratio_3_15_Eye >= Lower__3_15_Eye[i] ):
                H3_15_ID = True
                
            else:
                H3_15_ID = False
                
            
            if(Ratio_4_14_Eye <= Upper__4_14_Eye[i]  and  Ratio_4_14_Eye >= Lower__4_14_Eye[i] ):
                H4_14_ID = True
                
            else:
                H4_14_ID = False
                
            
            if(Ratio_5_13_Eye <= Upper__5_13_Eye[i]  and  Ratio_5_13_Eye >= Lower__5_13_Eye[i] ):
                H5_13_ID = True
                
            else:
                H5_13_ID = False
                
            
            if(Ratio_6_12_Eye <= Upper__6_12_Eye[i]  and  Ratio_6_12_Eye >= Lower__6_12_Eye[i] ):
                H6_12_ID = True
                
            else:
                H6_12_ID = False
                
            
            if(Ratio_7_11_Eye <= Upper__7_11_Eye[i]  and  Ratio_7_11_Eye >= Lower__7_11_Eye[i] ):
                H7_11_ID = True
                
            else:
                H7_11_ID = False
                
            
            if(Ratio_8_10_Eye <= Upper__8_10_Eye[i]  and  Ratio_8_10_Eye >= Lower__8_10_Eye[i] ):
                H8_10_ID = True
                
            else:
                H8_10_ID = False



    
            #Checking Face ID
    
            #print("First_ID =",First_ID)
            #print("Second_ID =", Second_ID)
            #print("Third_ID =", Third_ID)
    
    
    
            if(First_ID and Second_ID and Third_ID and Fourth_ID and Fifth_ID and Sixth_ID and Seventh_ID and Eighth_ID and L2_ID and L3_ID and L4_ID and L5_ID and L6_ID and L7_ID and L8_ID and R10_ID and R11_ID and R12_ID and R13_ID and R14_ID and R15_ID and R16_ID and H2_16_ID and H3_15_ID and H4_14_ID and H5_13_ID and H6_12_ID and H7_11_ID and H8_10_ID):
                print("You may be",filenames[i][:-4])
                count[i] = count[i] + 1
                print(count[i])
                #cv2.putText(frame, "You Are Shivam", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            else:
                #cv2.putText(frame, "You Are Not Shivam", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
                count[i] = 0

    
#            # Display the resulting frame
            if(count[i] >= 2):
                print("It is conform that You are",filenames[i][:-4])
                print("You are allowed to enter")
                count[i] = 0
                

#            if (filenames[i][:-4] == "Shivam"):
#                print("First_ID =",First_ID)
#                print("Second_ID =",Second_ID)
#                print("Third_ID =",Third_ID)
#                print("Fourth_ID =",Fourth_ID)
#                print("Fifth_ID =",Fifth_ID)
#                print("Sixth_ID =",Sixth_ID)
#                print("Seventh_ID =",Seventh_ID)
#                print("Eighth_ID =",Eighth_ID)
                
#                print("L2_ID =",L2_ID)
#                print("L3_ID =",L3_ID)
#                print("L4_ID =",L4_ID)
#                print("L5_ID =",L5_ID)
#                print("L6_ID =",L6_ID)
#                print("L7_ID =",L7_ID)
#                print("L8_ID =",L8_ID)
                
#                print("R10_ID =",R10_ID)
#                print("R11_ID =",R11_ID)
#                print("R12_ID =",R12_ID)
#                print("R13_ID =",R13_ID)
#                print("R14_ID =",R14_ID)
#                print("R15_ID =",R15_ID)
#                print("R16_ID =",R16_ID)
                
#                print("H2_16_ID =",H2_16_ID)
#                print("H3_15_ID =",H3_15_ID)
#                print("H4_14_ID =",H4_14_ID)
#                print("H5_13_ID =",H5_13_ID)
#                print("H6_12_ID =",H6_12_ID)
#                print("H7_11_ID =",H7_11_ID)
#                print("H8_10_ID =",H8_10_ID)
                
#                is_person = True
#            cv2.putText(frame, "You Are Shivam", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#        else:
#            cv2.putText(frame, "You Are Not Shivam", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
#    if (is_person):
#        cv2.putText(frame, "You Are Person", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#    else:
#        cv2.putText(frame, "Scanning for Person", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
