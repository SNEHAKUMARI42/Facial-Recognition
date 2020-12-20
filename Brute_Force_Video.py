import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('Faces/Shivam.jpg',0)          # queryImage

cap = cv2.VideoCapture(0)

while(ret):

    # Take each frame
    ret, frame = cap.read()
    #frame1 = frame.copy()


    img2 = frame.copy() # trainImage

    # Initiate SIFT detector
    orb = cv2.ORB_create()


    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(img1,None)
    kp2, des2 = orb.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)


    #print("des1",len(des1))
    #print("des2", len(des2))
    if ((des1 != None) and (des2 != None)):
        # Match descriptors.
        matches = bf.match(des1,des2)
        #print("matches",len(matches))

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)


        # Draw each and every matches.

        img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches,None, flags=2)
        #cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches1to2, outImg[, matchColor[, singlePointColor[, matchesMask[, flags]]]])
    else:
        img3 = img2

    cv2.imshow("img3",img3)
    # Press Esc to quit
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()