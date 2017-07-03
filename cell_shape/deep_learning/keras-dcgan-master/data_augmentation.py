# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 11:17:05 2017

@author: univ4208
"""
import glob
import os
from PIL import Image
import numpy as np
import cv2

savepath = r"D:\data_aurelien\data\deep_learning\cells"

def rotations(path):
    """given a dataset in path, operates 3 rotations of 90degres and
    saves them in the same path"""
    look_for = os.path.join(path,"*")
    arms_list = glob.glob(look_for)
    new_index = len(arms_list)
    
    for arm_name in arms_list:
        print new_index
        img = np.asarray(Image.open(arm_name))
        for i in range(3):
            img = np.rot90(img)
            cv2.imwrite(os.path.join(path,str(new_index)+".png"),img)
            new_index+=1
    return

def load_protrusions_list(path=r"D:\data_aurelien\data\deep_learning\arms_for_deep_learning"):
    
    print "Loading dataset..."
    look_for = os.path.join(path,"*")
    arms_list = glob.glob(look_for)
    first_arm = Image.open(arms_list[0])
    first_arm = np.asarray(first_arm)
    size = first_arm.shape[0]
    #Format: n_examples*width*height
    data = []
    for i,arm_name in enumerate(arms_list):
        img = np.asarray(Image.open(arm_name))
        data.append(img)
    print "done"
    return data


def shuffle_protrusions(path):
    dat = load_protrusions_list(savepath)
    t=dat[0]
    np.random.shuffle(dat)
    if np.all(t==dat[0]):
        print "shuffling failed"
    for i in range(len(dat)):
        img = dat[i]
        cv2.imwrite(os.path.join(path,str(i)+".png"),img)
        
def append_protrusions(path,new_path):
    dat = load_protrusions_list(path)
    look_for = os.path.join(new_path,"*")
    arms_list = glob.glob(look_for)
    new_index = len(arms_list)
    
    for i in range(len(dat)):
        img = dat[i]
        cv2.imwrite(os.path.join(new_path,str(new_index+i)+".png"),img)

path = r"D:\data_aurelien\data\deep_learning\cells2"
new_path = r"D:\data_aurelien\data\deep_learning\cells"
append_protrusions(path,new_path)