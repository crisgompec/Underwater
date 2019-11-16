#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

WHITE = 255
DARKEN_LAYER = 0

def inpaint(image, mask):
	RADIOUS = 3
	return cv2.inpaint(image,mask,RADIOUS,cv2.INPAINT_TELEA)

def reduce_mask(mask):

	N,M = np.shape(mask)
	WHITE = 255
	DARKEN_LAYER = 0

	for i in range(N):

		edge_detected = False
		for j in range(M):

			if mask[i][j] == WHITE and edge_detected == False:
				mask[i][j] = 0
				if j+DARKEN_LAYER<M:
					for k in range(1,DARKEN_LAYER+1):
						mask[i][j+k]=0

				if j+DARKEN_LAYER+1 == WHITE:
					edge_detected = True

			elif mask[i][j] == 0 and edge_detected == True:
				mask[i][j-1] = 0
				edge_detected = False
				if j-DARKEN_LAYER>=0:
					for k in range(1,DARKEN_LAYER+1):
						mask[i][j-k]=0



if __name__ == "__main__":
	image = cv2.imread('temp_images/result.png') #'images/UW_400.png'
	mask = cv2.imread('temp_images/mask.png')
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	#reduce_mask(mask)
	output = inpaint(image, mask)
	cv2.imwrite('temp_images/inpainted_image_result.png', output)

