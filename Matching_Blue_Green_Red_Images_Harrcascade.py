# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:33:18 2019

@author: Shivam
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 11:07:18 2019

@author: Shivam
"""

from imutils import face_utils
import dlib
import cv2
import numpy as np
import math

img = cv2.imread("Faces/Shivam.jpg")
b_img,g_img,r_img = cv2.split(img)


face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)

    if faces is():
        return None

    for(x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]


    return cropped_face


cap = cv2.VideoCapture(0)
cv2.waitKey(4)

ret, frame = cap.read()

while(ret):
    # Capture frame-by-frame
    ret, frame = cap.read()
    frame1 = frame.copy()
    
    if face_extractor(frame1) is not None:
        
        face = cv2.resize(face_extractor(frame1),(200,200))
        
        
        b_video = face[:,:,0]
        g_video = face[:,:,1]
        r_video = face[:,:,2]
        
    
        # Initiate SIFT detector
        orb = cv2.ORB_create()
    
    
        # find the keypoints and descriptors with SIFT Blue
        kp1_b, des1_b = orb.detectAndCompute(b_img,None)
        kp2_b, des2_b = orb.detectAndCompute(b_video,None)
        
        # find the keypoints and descriptors with SIFT Green
        kp1_g, des1_g = orb.detectAndCompute(g_img,None)
        kp2_g, des2_g = orb.detectAndCompute(g_video,None)
    
        # find the keypoints and descriptors with SIFT Red
        kp1_r, des1_r = orb.detectAndCompute(r_img,None)
        kp2_r, des2_r = orb.detectAndCompute(r_video,None)
    
    
    
        # create BFMatcher object Blue
        bf_b = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        # create BFMatcher object Green
        bf_g = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # create BFMatcher object Red
        bf_r = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
        #print("des1",len(des1))
        #print("des2", len(des2))
        
        #Blue Video
        if ((des1_b != None) and (des2_b != None)):
            # Match descriptors.
            matches_b = bf_b.match(des1_b,des2_b)
            #print("matches",len(matches))
    
            # Sort them in the order of their distance.
            matches_b = sorted(matches_b, key = lambda x:x.distance)
    
    
            # Draw each and every matches.
    
            frame_b = cv2.drawMatches(b_img,kp1_b,b_video,kp2_b,matches_b,None, flags=2)
            #cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])
        else:
            frame_b = b_video
    
        cv2.imshow("Blue",frame_b)
        cv2.waitKey(1)
        
        #Green Video
        if ((des1_g != None) and (des2_g != None)):
            # Match descriptors.
            matches_g = bf_g.match(des1_g,des2_g)
            #print("matches",len(matches))
    
            # Sort them in the order of their distance.
            matches_g = sorted(matches_g, key = lambda x:x.distance)
    
    
            # Draw each and every matches.
    
            frame_g = cv2.drawMatches(g_img,kp1_g,g_video,kp2_g,matches_g,None, flags=2)
            #cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])
        else:
            frame_g = g_video
    
        cv2.imshow("Green",frame_g)
        cv2.waitKey(1)
        
        
        #Red Video
        if ((des1_r != None) and (des2_r != None)):
            # Match descriptors.
            matches_r = bf_b.match(des1_r,des2_r)
            #print("matches",len(matches))
    
            # Sort them in the order of their distance.
            matches_r = sorted(matches_r, key = lambda x:x.distance)
    
    
            # Draw each and every matches.
    
            frame_r = cv2.drawMatches(r_img,kp1_r,r_video,kp2_r,matches_b,None, flags=2)
            #cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])
        else:
            frame_r = r_video
    
        cv2.imshow("Red",frame_r)
        cv2.waitKey(1)

        
        
        #face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        #file_name_path = 'C:/opencv/faces/user'+str(count)+'.jpg'
        #cv2.imwrite(file_name_path,face)

        #cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
        cv2.imshow("Face Cropper",face)

    else:
        print("Face not Found")
        pass
    
    
        
    
    if cv2.waitKey(1) &0xFF == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
