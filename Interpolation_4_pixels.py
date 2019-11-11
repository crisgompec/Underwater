#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import dippykit as dip
import matplotlib.pyplot as plt
import pandas as pd


#PICKING AN IMAGE
im_original = cv2.imread("images/UW_400.png")
im_original = cv2.imread("images/underwater-15_downsampled.png")
#im_original = cv2.imread("images/underwater-15.png")


#===========================================
#INITIAL EXPLORATION OF THE PICTURE
#Select pixels that are similar in color to skin
im_skin = np.copy(im_original)
for i in range(np.shape(im_original)[0]):
    for j in range(np.shape(im_original)[1]):
        #if im_original[i,j][2] < 60:
        if im_original[i,j][2] < 60 or im_original[i,j][0] > 200: #Zones mostly reddish, not blueish
            im_skin[i,j]=[0,0,0]

cv2.imwrite("temp_images/skin_color.png", im_skin)

#Select zones with high luminosity (probably zones iluminated by refracted light)
im_gray = cv2.cvtColor(im_original, cv2.COLOR_BGR2GRAY)
ret,th1 = cv2.threshold(im_gray,180,255,cv2.THRESH_BINARY)

cv2.imwrite("temp_images/binary.png", th1)

#Saliency detection
# initialize OpenCV's static fine grained saliency detector and compute the saliency map
saliency = cv2.saliency.StaticSaliencyFineGrained_create()
(success, saliencyMap) = saliency.computeSaliency(im_original)

# if we would like a *binary* map that we could process for contours,
# compute convex hull's, extract bounding boxes, etc., we can
# additionally threshold the saliency map

threshMap = cv2.threshold(saliencyMap*255, 80, 255,cv2.THRESH_BINARY)[1]

cv2.imwrite("temp_images/saliency.png", saliencyMap*255)

#=============================================
#CALCULATE AVERAGE SKIN COLOR         
aux_skin_value = [0,0,0]
num_pixels_for_skin = 0

for i in range(np.shape(im_skin)[0]):
    for j in range(np.shape(im_skin)[1]):
        aux_skin_value = aux_skin_value + im_skin[i,j]
        if (im_skin[i,j] > [0, 0, 0]).all():
            num_pixels_for_skin = num_pixels_for_skin + 1
            
avg_skin_color = aux_skin_value/num_pixels_for_skin

print('The average skin color (RGB) is:')
print(avg_skin_color)

#CALCULATE AVERAGE SKIN COLOR (ONLY BRIGHTER AREAS)
#Make an histogram with the color of the pixels
im_skin_vec = np.reshape(im_skin, (np.shape(im_skin)[0]*np.shape(im_skin)[1],3))
df = pd.DataFrame(im_skin_vec, columns=['R', 'G', 'B']) 

df['Intensity'] = df.apply(lambda row: (row.R + row.G + row.B)/3, axis = 1) 

df_color = df[df['Intensity'] > 0]

print(df_color)

"""
%matplotlib inline
df_color.plot.hist(bins = 50)

%matplotlib inline
plot = df_color.Intensity.plot(kind='kde')

"""

df_brigth_color = df[(df['Intensity'] > 70) & (df['Intensity'] < 87)]

sum_rgb = df_brigth_color.sum(axis=0)
average_rgb = sum_rgb/len(df_brigth_color)

c = 1.15 #For making the color brighter




avg_skin_color = (average_rgb[0]*c, average_rgb[1]*c, average_rgb[2]*c)

print('The average skin color (RGB) for brighter areas is:') 
print(avg_skin_color)


#INTERPOLATION

im_skin_interp = np.copy(im_skin)

#Step 1: Go over all the pixels of "Zones similar to skin color" using a square of 4 pixels

mask = np.zeros((np.shape(im_skin)[0], np.shape(im_skin)[1]))

for i in range(np.shape(im_skin)[0] - 1):
    for j in range(np.shape(im_skin)[1] -1):
        
        pixel_upper_left = im_skin[i,j]
        pixel_upper_rigth = im_skin[i,j+1]
        pixel_bottom_left = im_skin[i+1 ,j]
        pixel_bottom_rigth = im_skin[i+1,j+1]
        
        pixel_array = [pixel_upper_left, pixel_upper_rigth, pixel_bottom_left, pixel_bottom_rigth]
        
        remove_pixel = [False, False, False, False]
        
                
        #Step 2: Check if at least one of the 4 pixels correspond to the body (i.e. it isn´t black)
        
        if (pixel_upper_left > ([0, 0, 0])).all() or (pixel_upper_rigth > ([0,0,0])).all or (pixel_bottom_left > ([0,0,0])).all() or (pixel_bottom_rigth > ([0,0,0])).all():
            
            #Step 3: Check if any of those pixels are marked as pixels to remove (i.e. they are white on image after binary threshold)
            if th1[i,j] == 255: 
                remove_pixel[0] = True
            if th1[i,j+1] == 255: 
                remove_pixel[1] = True
            if th1[i+1,j] == 255: 
                remove_pixel[2] = True
            if th1[i+1,j+1] == 255: 
                remove_pixel[3] = True
            
            #We need to remove the targeted pixels and substitute its value with the average of the surrounding pixels that are part of the body
            
            aux_pixel_value = [0,0,0]
            num_pixels_for_average = 0
            
            for p in range(np.shape(pixel_array)[0]):
                #if remove_pixel[p] == False and (pixel_array[p] > ([0, 0, 0])).all(): 
                if remove_pixel[p] == False: #The image has less false skin-coloured pixels
                    aux_pixel_value = aux_pixel_value + pixel_array[p]
                    num_pixels_for_average = num_pixels_for_average + 1
                    
            if num_pixels_for_average > 0: 
                avg_pixel_value = aux_pixel_value/num_pixels_for_average
            elif num_pixels_for_average == 0:
                avg_pixel_value = avg_skin_color #FOR BRIGHTER AREAS

                mask[i][j] = 255
                mask[i+1][j] = 255
                mask[i][j+1] = 255
                mask[i+1][j+1] = 255
                
            
            for p in range(np.shape(pixel_array)[0]):
                if remove_pixel[p] == True: 
                    pixel_array[p] =  avg_pixel_value
                    
            

            if (im_skin[i,j] == [0,0,0]).all():
                im_skin_interp[i,j] = pixel_array[0]
            if (im_skin[i,j+1] == [0,0,0]).all():
                im_skin_interp[i,j+1] = pixel_array[1]
            if (im_skin[i+1,j] == [0,0,0]).all():
                im_skin_interp[i+1 ,j] = pixel_array[2]
            if (im_skin[i+1,j+1] == [0,0,0]).all():
                im_skin_interp[i+1,j+1] = pixel_array[3]
            
#Display only skin-coloured pixels on original image and interpolated skin 

"""
%matplotlib inline
dip.imshow(im_skin)
dip.show()

%matplotlib inline
dip.imshow(im_skin_interp)
dip.show()

"""
#COPY PIXELS OF INTERPOLATED SKIN ONTO THE ORIGINAL PICTURE

im_res = np.copy(im_original)


cv2.imwrite('temp_images/mask.png',mask)

"""
%matplotlib inline
dip.imshow(im_original)
dip.show()

"""
# Until here im_skin_interp[i,j] is the interpolated image with black background.



for i in range(np.shape(im_res)[0]):
    for j in range(np.shape(im_res)[1]):
        if (im_skin_interp[i,j] > [0, 0, 0]).all():
            im_res[i,j] = im_skin_interp[i,j]
"""            
%matplotlib inline
dip.imshow(im_res)
dip.show()
"""
cv2.imwrite("temp_images/result.png", im_res)
