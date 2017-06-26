# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:14:29 2017

@author: univ4208
"""

import numpy as np
from PIL import Image
import glob
import math
import os
import matplotlib.pyplot as plt
plt.close('all')
savepath = r"D:\data_aurelien\data\deep_learning\arms_for_deep_learning"

def load_n_protrusions(path=r"D:\data_aurelien\data\deep_learning\arms_for_deep_learning",n=128):
    
    print "Loading dataset..."
    look_for = os.path.join(path,"*")
    arms_list = glob.glob(look_for)
    first_arm = Image.open(arms_list[0])
    first_arm = np.asarray(first_arm)
    size = first_arm.shape[0]
    #Format: n_examples*width*height
    data = np.zeros( (n,size,size),dtype=np.uint8 )  
    indices = ( np.random.rand(n)*len(arms_list) ).astype(np.int)
    for j,i in enumerate(indices):
        img = np.asarray(Image.open(arms_list[i]))
        data[j,:,:]=img
    print "done loading",data.shape[0],"images"
    return data

def combine_images(generated_images):
    num = generated_images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num)/width))
    shape = generated_images.shape[1:]
    image = np.zeros((height*shape[0], width*shape[1]),
                     dtype=generated_images.dtype)
    for index, img in enumerate(generated_images):
        i = int(index/width)
        j = index % width
        image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
            img[ :, :]
    return image

dat = load_n_protrusions()
cb = combine_images(dat)
plt.figure()
plt.imshow(cb,cmap="gray")