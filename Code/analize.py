#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import dippykit as dip
import cv2


def display_images(images, text):
	num = len(images)

	dip.figure()
	for i in range(num):
		dip.subplot(1,num,i+1)
		dip.title(text[i])
		dip.imshow(images[i])

	dip.show()

def display_diff(im1,im2):
	im_diff = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY) - cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
	dip.figure()
	dip.imshow(im_diff)
	dip.title('Difference image')
	dip.show()

def iqa_results(im1,im2):
	print('The MSE is: {:.2f}'.format(dip.MSE(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY),cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY))))
	print('The PSNR is: {:.2f}'.format(dip.PSNR(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY),cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY),np.amax(cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)))))


if __name__ == "__main__":
	original = dip.im_read('images/original.png')
	result = dip.im_read('temp_images/result.png')
	display_images([original, result], ['Original image','Result image'])
	display_diff(original,result)
	iqa_results(original, result)

