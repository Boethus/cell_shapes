from skimage.morphology import watershed
import numpy as np
import cv2

from scipy import ndimage as ndi

from skimage import color
import scipy.ndimage as ndimage

import skimage.filters
from skimage.feature import peak_local_max
from skimage.morphology import watershed

import cv2
import scipy.misc

def filter_by_size(img_segm):
    """filters a segmented image by getting rid of the components with too few pixels"""
    
    numbers = np.zeros(np.max(img_segm-1))
    for i in range(1,np.max(img_segm)):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm))
    indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered

def canny_threshold(image):
    """Performs thresholding of an image using contour detection. This algorithm can be decomposed in two steps:
    1/Canny filter to get only edges in the image
    2/Gaussian blur
    3/Contour detection and filling"""
    size = 3
    
    #Step 1 
    canny = skimage.feature.canny(image)
    canny = canny/np.max(canny)*240     #!!Very important otherwise fails miserably. Flagadagada
    canny = canny.astype(np.uint8)
    #Step 2
    canny_blur = cv2.blur(canny,(size,size))
    canny_blur,contour,hierarchy = cv2.findContours(canny_blur,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    #Step 3
    i=0
    img = np.zeros(image.shape)
    for cnt in contour:
            cv2.drawContours(img,contour,i,255,-1)
            i+=1
    return img

def watershed_for_cells(gray,thresh):

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    opening = opening.astype(np.uint8)
    
    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.cv.CV_DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    markers, nLabels = ndi.label(sure_fg)

    #ret, markers = cv2.connectedComponents(sure_fg)   Not available with opencv 2.4
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==np.max(unknown)] = 0

    markers = markers.astype(np.int32)
    gray_converted = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.watershed(gray_converted,markers)
    gray_converted[markers == -1] = [255,0,0]
    return gray_converted,markers

