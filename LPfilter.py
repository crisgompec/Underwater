#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dippykit as dip
from math import exp
import numpy as np

# Proof of concept low pass filter
dim_filter = 800

h = np.zeros((dim_filter,dim_filter))
for u in range(dim_filter):
	for v in range(dim_filter):
		h[u][v] = exp(-(u+v)/(dim_filter*0.3))


# Loading image
im = dip.im_read('images/UW_400.png')
im = dip.im_to_float(im)

if 2 < im.ndim:
    im = np.mean(im, axis=2)

F = dip.fft2(im)
print(h*F)

#Plot results
#Original spectra
dip.figure(1)

dip.subplot(2, 2, 1)
dip.imshow(im, 'gray')
dip.title('Original Image')

dip.subplot(2, 2, 2)
dip.imshow(np.real(dip.ifft2(F*h)), 'gray')
dip.title('Modified image')

dip.subplot(2, 2, 3)
dip.imshow(abs(np.log(dip.fftshift(F))), 'gray')
dip.title('Original spectra')

dip.subplot(2, 2, 4)
dip.imshow(abs(np.log(dip.fftshift(F)*h)), 'gray')
dip.title('Modified spectra')

dip.show()

