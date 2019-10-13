# -*- coding: utf-8 -*-
"""
This script has the following functionalitites:
    - Threshold to separate people from the water surroundings
    - Usage of saliency detection to make a set of pixels that belongs to the person
"""
import cv2
import numpy as np
import dippykit as dip
import matplotlib.pyplot as plt


#PICKING AN IMAGE
im_original = cv2.imread("images/underwater-15.png")
#cv2.imshow("Original image", im_original)

#Select pixels that are similar in color to skin
im_skin = np.copy(im_original)
for i in range(np.shape(im_original)[0]):
    for j in range(np.shape(im_original)[1]):
        if im_original[i,j][2] < 60:
            im_skin[i,j]=[0,0,0]

#cv2.imshow("Zones similar to skin color", im_skin)
# record the results
cv2.imwrite("temp_images/skin_color.png", im_skin)

#Select zones with high luminosity (probably zones iluminated by refracted light)
im_gray = cv2.cvtColor(im_original, cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(im_gray,180,255,cv2.THRESH_BINARY)
#cv2.imshow("Image after Binary Threshold", th1)
# record the results
cv2.imwrite("temp_images/binary.png", th1)

#Saliency detection
# initialize OpenCV's static fine grained saliency detector and compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(im_original)
#cv2.imshow("Output of Saliency Detection", saliencyMap)
# record the results
cv2.imwrite("temp_images/saliency.png", saliencyMap*255)

# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map

threshMap = cv2.threshold(saliencyMap*255, 60, 255,
cv2.THRESH_BINARY)[1]
#cv2.imshow("Thresh", threshMap)
# record the results
cv2.imwrite("temp_images/saliency_threshold.png", threshMap)
