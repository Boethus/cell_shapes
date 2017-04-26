# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 18:14:23 2017

@author: aurelien
"""

import numpy as np
import cv2
import os
import imageio
import matplotlib.pyplot as plt
import functions as f

from skimage import color
import scipy.ndimage as ndimage

from skimage.filter import threshold_otsu
import skimage.filters
import newFunctions as nf

import scipy.misc
save=False
plt.close('all')

def sh(im):
    plt.figure()
    plt.imshow(im,cmap='gray')
    plt.colorbar()

os.chdir('/home/aurelien/Documents/Oxford/rotation_Rittscher/')
filename = 'itoh-cell-migration-02.mov'
vid = imageio.get_reader(filename,  'ffmpeg')
img = vid.get_data(1)
img = img[30:,:]

vid_len = vid.get_length()

img = np.array(img)
img = color.rgb2gray(img)

if save:
    scipy.misc.imsave('sample_image.png', img)

edge_filtered = skimage.filters.edges.sobel(img)
filt_otsu = threshold_otsu(edge_filtered)
binary_otsu = edge_filtered>filt_otsu
sh(edge_filtered)
print np.min(edge_filtered),np.max(edge_filtered)

thresh = skimage.filters.median(edge_filtered)
#edgy = scipy.misc.imread('sample_image_edge.png')
sh(thresh)

"""
grey_im=cv2.cvtColor(img[50:,:], cv2.COLOR_RGB2GRAY )
grey_im = np.invert(grey_im)
grey_im = f.gaussian_convolution(grey_im,5)
grey_im = grey_im.astype(np.uint8)

ret,thresh = cv2.threshold(grey_im,0,255,cv2.THRESH_OTSU)
print type(thresh[0,0]),thresh.shape, "snooop threshold"
plt.figure()
plt.imshow(thresh)


segm,nb = f.watershed_segmentation(grey_im,thresh)
"""
