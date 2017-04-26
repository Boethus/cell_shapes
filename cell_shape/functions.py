# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:06:54 2016

@author: pi
"""

# Standard imports
import cv2
import numpy as np
from scipy import ndimage
from scipy import signal
#import scipy.signal as sg
import matplotlib.pyplot as plt
import identifier

#Model parameters
minimum_size = 3000
maximum_size = 30000
gaussian_kernel_size = 5 #in pixels
value_threshold=240
coins = ["2po","1po","50pe","20pe","10pe","5pe","2pe","1pe"]

def gaussian_convolution(array,size):
    """Method for blurring an image with a gaussian kernel
    
    :param np.ndarray array: 2D array to convolve
    :param int size: size of the gaussian kernel 
    :return:
        np.ndarray out: image convolved
    """
    x,y=np.ogrid[-size//2:size//2,-size//2:size//2]
    d2=x**2+y**2
    d2 = np.exp(-d2/(size/2)**2 )
    d2/=np.sum(d2)  #normalization
    out = signal.convolve2d(array,d2,mode="same")
    return out

    
def fillHoles(thresh):
    """Fils the holes in a thresholded image, basing on contours 
    detection and filling.
    
    :param np.ndarray thresh: 2D array (type uint8, values =0 or 255)
    :return:
        np.ndarray image: image with no holes"""
    image,contour,hier = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)

    i=0
    for cnt in contour:
        cv2.drawContours(image,contour,i,255,-1)
        i+=1
    return image
    
def centroid(array,value):
    """Computes the centroid of a certain group of points in a segmented image
    
    :param np.ndarray array: 2D segmented image. All points of an identical 
        segment have the same value
    :param int value: value of the points corresponding to the segment of interest
    :return:
        list coordinates: coordinates of the centroid
        """
    coordinates = np.where(array==value)
    number_of_points = len(coordinates[0])
    x = np.sum(coordinates[0])/number_of_points
    y = np.sum(coordinates[1])/number_of_points
    return [x,y]
    
def watershed_segmentation(img,thresh):
    """Watershed segmentation algorithm.
    
    :param np.ndarray img: image to be segmented
    :param np.ndarray thresh: thresholded image
    :return:
        np.ndarray markers: image segmented
        int nr_objects: number of objects detected"""
    thresh=thresh/np.max(thresh)
    thresh=thresh.astype(np.uint8)
    kernel = np.ones((3,3), np.uint8)
    #Dilate to find area that is 100% sure background
    sure_bg = cv2.dilate(thresh, kernel, iterations=3)
    
    #Erode to find are that pertains 100% sure to coins
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    ret, sure_coins = cv2.threshold(dist_transform, 0.01*dist_transform.max(), 255, 0)
    
    #Find indetermined area that contains borders
    sure_coins = np.uint8(sure_coins)
    indetermined = cv2.subtract(sure_bg, sure_coins)
                                                     
    #Marker labelling
    ret, markers = cv2.connectedComponents(sure_coins)
    
    #Add one to all labels so that sure background is not 0 but 1.
    markers = markers + 1
    print type(markers[0,0]),np.max(markers),markers.shape
    #Mark the region of unknown with zero
    markers[indetermined == 255] = 0
    markers = markers.astype(np.int32)
    img = img.astype(np.int32)
    print type(img[0,0]), type(markers[0,0]), 'snoop doggi dog'
    markers2 = cv2.watershed(img, markers)
    nr_objects = np.max(markers2)
    return markers2,nr_objects
    
def labelPoint(array,point):
    """Displays a square of side size 5 at the location point in array.
    
    :param np.ndarray array: image to be labeled
    :param list point: 2-elements list containing the coordinates 
        of the point to label"""
    size=5
    #use these min and max variables to avoid indexes out of range in the edges
    min_x=max(0,point[0]-size)
    min_y=max(0,point[1]-size)
    max_x=min(array.shape[0],point[0]+size)
    max_y=min(array.shape[1],point[1]+size)
    array[min_x:max_x,min_y:max_y]=np.ones((max_x-min_x,max_y-min_y))*5.5
    
def area(array,value):
    """Computes the area (in pixels) of an object in a segmented image.
    
    :param np.nadrray array: segmented image
    :param int value: value of the group of the pixels of interest.
    :return:
        int area: the area of value in array, in pixels
    """
    val = np.zeros(array.shape)
    val[array==value]=1
    return np.sum(val)

def coinsFromLabels(labeled_image,coin_labels):
    """Returns an image labeling only the coins from a segmented image with different 
    labels and the list of labels corresponding to coins

    :param np.ndarray labeled_image: segmented image    
    :param list coin_labels: list containing the labels of the coins
    :return:
        np.ndarray new_image: image containing only the coins
    """
    
    new_image = np.zeros(labeled_image.shape)
    for u in range(len(coin_labels)):
        new_image[np.where(labeled_image==coin_labels[u])]=u+1
    return new_image

def findObjects(image):
    """
    Finds coins in a bgr image.
    
    :param np.ndarray image: 3D array, BGR image
    :return: 
        list centroid:  list containing the coordinates of 
        the centres of the different objects detected in the image.
        np.ndarray labeled: the segmented array containing the position of the 
        different objects
        list coin_labels: a list containing the values of the objects corresponding
        to a coin
        list areas: contains the areas of the corresponding coins
    """
    grey_im=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )
    #Gaussian smoothing to eliminate the artefacts
    grey_im = gaussian_convolution(grey_im,gaussian_kernel_size)
    grey_im = grey_im.astype(np.uint8)
    ret,thresh = cv2.threshold(grey_im,0,255,cv2.THRESH_OTSU)

    thresh = np.invert(thresh)
    thresh = fillHoles(thresh)
    #labeled,nr_objects = ndimage.label(thresh>value_threshold)
    labeled,nr_objects = watershed_segmentation(image,thresh)
    centroids=[]
    coin_labels=[]
    actual_number_objects =0
    areas=[]

    print nr_objects, "objects found"
    for i in range(nr_objects):
        size=area(labeled,i+1)
        if(size>minimum_size and size<maximum_size):
            #Determining the minimum enclosing circle to get circular coins
            points = np.where(labeled==i+1)
            points_vector = np.zeros( (len(points[0]),2) )
            points_vector[:,1] = points[0]
            points_vector[:,0] = points[1]
            points_vector=points_vector.astype(int)
            actual_number_objects+=1
            c=centroid(labeled,i+1)
            centroids.append(c)
            coin_labels.append(i+1)
            areas.append(size)
            
            #Set new value to labeled to only label the coins
  
    return centroids,labeled,coin_labels,areas

def identify_from_size(size,colour):
    """Indentifies a coin from its size and colour once located and its colour 
    determined
    
    :param int size: area in pixels of the coin
    :param float colour: mean Saturation value of the coin
    :return:
        str coin: the name of the coin detected"""
    sizes = np.load("coin_areas.npy")

    yellow_areas=sizes[0:2]
    grey_areas = sizes[2:6]
    red_areas = sizes[6:8]
    #Finding coin has the area closest from size
    if colour=="grey":
        grey_areas = [abs(x-size) for x in grey_areas]
        index = 2+ grey_areas.index(min(grey_areas))
        return coins[index]
    if colour=="red":
        red_areas = [abs(x-size) for x in red_areas]
        index = 6+ red_areas.index(min(red_areas))
        return coins[index]
    if colour=="yellow":
        yellow_areas = [abs(x-size) for x in yellow_areas]
        index = yellow_areas.index(min(yellow_areas))
        return coins[index]
