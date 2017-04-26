# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 16:06:54 2016

@author: pi
"""

# Standard imports
import cv2
import numpy as np
import os
from scipy import ndimage
from scipy import signal
#import scipy.signal as sg
import matplotlib.pyplot as plt

folder_to_samples = "inFocusCoins"

def computeScore(array,ref,point):
    """Computes the scalar product of a reference array with a specific position 
    inside an image
    
    :param np.ndarray array: the image in which we look for coins
    :param np.ndarray ref: the reference image.
    :param list point: the coordinates of the coin to test
    :return:
        float score: scalar product, the higher, the best the two templates match
    """
    sizex = ref.shape[0]
    sizey = ref.shape[1]
    #use these min and max variables to avoid indexes out of range in the edges
    min_x=max(0,point[0]-sizex//2)
    min_y=max(0,point[1]-sizey//2)
    max_x=min(array.shape[0],point[0]+sizex-sizex//2)
    max_y=min(array.shape[1],point[1]+sizey-sizey//2)
    
    shape_in_array = array[min_x:max_x,min_y:max_y]
    if sizex>shape_in_array.shape[0]:
        if min_x==0:
            ref=ref[(sizex-shape_in_array.shape[0]):,:]
        else:
            ref=ref[:-(sizex-shape_in_array.shape[0]),:]
    if sizey>shape_in_array.shape[1]:
        if min_y==0:
            ref=ref[sizey-shape_in_array.shape[1]:,:]
        else:
            ref=ref[:-(sizey-shape_in_array.shape[1]),:]
    
    norm_shape = np.sqrt(np.sum(shape_in_array**2))
    norm_ref = np.sqrt(np.sum(ref**2))
    score = np.sum(shape_in_array*ref)*10000
    score/=(norm_ref)
    score/=norm_shape
    return score
    
def identify(array,point):
    """Identifies the object at the position coordinates. Tries every templates
    available and determines which one has the closest shape
    
    :param np.ndarray array: image
    :param list point: location of the coin to test.
    :return:
        str template: the name of the template yielding the best match"""
    templates = os.listdir(folder_to_samples)  #Getting all the coin templates we have
    nb_templates = len(templates)
    scores=np.zeros(nb_templates) #For each coin template, compute a score, the higher the closer
    for i in range(nb_templates):
        img = cv2.imread(folder_to_samples+"/"+templates[i], 0)
        scores[i]=computeScore(array,img,point)
    #scores=scores[:-1] #removing the 20pe which somehow blasts everything
    index_max_score = scores.argmax()
    return templates[index_max_score]
    
def identifyColor(array,point):
    """Identifies the object at the position coordinates, taking into account the
    
    :param np.ndarray array: image
    :param list point: location of the coin to test.
    :return:
        str template: the name of the template yielding the best match"""
    templates = os.listdir(folder_to_samples)  #Getting all the coin templates we have

    nb_templates = len(templates)
    scores=np.zeros((nb_templates,3)) #For each coin template, compute a score, the higher the closer
    for j in range(3):
        for i in range(nb_templates):
            img = cv2.imread(folder_to_samples+"/"+templates[i])
       #     img = cv2.equalizeHist(img)
            
            scores[i,j]=computeScore(array[:,:,j],img[:,:,j],point)
    print "scores", scores
    print templates
    scores=np.sum(scores,axis=1)
    index_max_score = scores.argmax()

    return templates[index_max_score]    

    
