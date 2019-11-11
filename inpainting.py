#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import cv2

def inpaint(image, mask):
	RADIOUS = 3
	return cv2.inpaint(image,mask,RADIOUS,cv2.INPAINT_TELEA)


if __name__ == "__main__":
	image = cv2.imread('temp_images/result.png') #'images/UW_400.png'
	mask = cv2.imread('temp_images/mask.png')
	mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
	output = inpaint(image, mask)
	cv2.imwrite('temp_images/inpainted_image_result.png', output)

