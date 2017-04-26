# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:48:05 2017

@author: univ4208
"""
import numpy as np
import cv2
import os

def showSegmentation_color(filename,total_buffer,save=False,savename='video8'):
    """Based on a pre-established segmentation shows the corresponding movie"""
    cap = cv2.VideoCapture(os.path.join("data",filename))
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    max_index = np.amax(total_buffer)
    colors = []
    total_buffer = total_buffer.astype(np.uint8)

    for i in range(max_index):
        color_intensity = (max_index+1)//3 * (i//3+1) *40
        if i%3 == 0:
            colors.append((color_intensity,i//3 * color_intensity,abs(1-i//3)*color_intensity))
        if i%3 == 1:
            colors.append((abs(1-i//3)*color_intensity,
                           color_intensity,color_intensity,
                           i//3 * color_intensity))
        if i%3 == 2:
            colors.append((i//3 * color_intensity,
                           abs(1-i//3)*color_intensity,
                           color_intensity))
    for i in range(length):
        #Get frame
        ret, frame = cap.read()
        frame=frame[30:,:,:]
        thresh = total_buffer[:,:,i]
        thresh = np.copy(thresh)
        drawCentroids2(frame,thresh,colors)
        thresh,contour,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contour, -1, (0,255,0), 3)
        
        cv2.imshow('frame',frame)
        cv2.waitKey(100)
        if save:
            cv2.imwrite(os.path.join("data",savename,'frame'+str(i)+'.png'),frame )

def showSegmentation(filename,total_buffer,save=False,savename='video7'):
    """Based on a pre-established segmentation shows the corresponding movie"""
    cap = cv2.VideoCapture(os.path.join("data",filename))
    
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in range(length):
        #Get frame
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray=gray[30:,:]
        gray_contours = drawContours(gray,total_buffer[:,:,i])
        
        cv2.imshow('frame',gray_contours)
        cv2.waitKey(40)
        if save:
            cv2.imwrite(os.path.join("data",savename,'frame'+str(i)+'.png'),total_buffer[:,:,i])
            
def drawCentroids(image,xs,ys):
    for i in range(len(xs)):
        cv2.circle(image, (int(ys[i]),int(xs[i])), 10, (0,0,255), -1)

def drawCentroids2(image, img_segm,colors):
    m = np.max(img_segm)
    xs = []
    ys = []
    elts = []
    for i in range(0,m):
        if np.any(img_segm==i+1):
            pos_list = np.where(img_segm==i+1)
            xs.append(np.mean(pos_list[0]))
            ys.append(np.mean(pos_list[1]))
            elts.append(i)
    for i in range(len(xs)):
        cv2.circle(image, (int(ys[i]),int(xs[i])), 10, colors[elts[i]], -1)
        
        
def drawContours(img,thresh):
    thresh = thresh.astype(np.uint8)
    thresh,contour,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    image = np.copy(img)
    cv2.drawContours(image, contour, -1, (0,255,0), 3)
    return image