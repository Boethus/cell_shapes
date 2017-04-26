# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 10:31:13 2017

@author: aurelien
"""

from skimage.util import img_as_ubyte
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filter import threshold_otsu
import numpy as np
import matplotlib.pyplot as plt

def waterShed(img):
    """Applies watershed algorithm to greyscale image img"""
    img_u = img_as_ubyte(img)
    plt.figure()
    plt.imshow(img_u)
    plt.colorbar()
    #Compute the distance map on the binarised image
    block_size = 30
    otsu_thresh = threshold_otsu(img_u)
    binary_otsu = img_u > otsu_thresh
    
    distance = ndi.distance_transform_edt(binary_otsu)
    plt.figure()
    plt.imshow(binary_otsu)
    plt.colorbar()
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)), labels=img)
    markers = ndi.label(local_maxi)[0]
    
    labels = watershed(-distance, markers, mask=binary_otsu)
    return labels