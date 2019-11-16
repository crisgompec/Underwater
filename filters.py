#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import argparse

def median_filter(image):
	# create the argument parser and parse the arguments
	# ap = argparse.ArgumentParser()
	# ap.add_argument('-i', '--image', required = True, help = 'Path to the input image')
	# args = vars(ap.parse_args())
	filter_window = 3

	# apply the median filter on the image
	processed_image = cv2.medianBlur(image, filter_window)
	# display image
	cv2.imshow('Median Filter Processing', processed_image)
	# save image to disk
	cv2.imwrite('temp_images/medianfilter_image_result.png', processed_image)
	# pause the execution of the script until a key on the keyboard is pressed
	cv2.waitKey(0)


if __name__ == "__main__":
	image = cv2.imread('temp_images/result.png') #'images/UW_400.png'
	median_filter(image)



