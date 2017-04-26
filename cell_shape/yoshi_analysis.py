#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 10:56:25 2017

@author: aurelien
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from scipy import ndimage as ndi

from skimage import color
import scipy.ndimage as ndimage

from skimage.filters import threshold_otsu
import skimage.filters

import scipy.misc
import time

import segmentationFunctions as sf
import hungarian

from sklearn.neighbors import KNeighborsClassifier

def filter_by_size(img_segm):
    """filters a segmented image by getting rid of the components with too few pixels"""
    
    numbers = np.zeros(np.max(img_segm-1))
    for i in range(1,np.max(img_segm)):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm))
    indexes = indexes[numbers>2*np.mean(numbers)] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    segm_filtered=segm_filtered.astype(np.int)
    return segm_filtered

def drawContours(img,thresh):
    thresh = thresh.astype(np.uint8)
    thresh,contour,hierarchy = cv2.findContours(thresh,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    image = np.copy(img)
    cv2.drawContours(image, contour, -1, (0,255,0), 3)
    return image
            
def centroids(img_segm):
    """Computes the positions of centroids in a segmented image.
    Returns the centroids numbers in order"""
    m = np.amax(img_segm)
    xs = np.zeros(m)
    ys = np.zeros(m)
    
    for i in range(0,m):
        pos_list = np.where(img_segm==i+1)
        xs[i] = np.mean(pos_list[0])
        ys[i] = np.mean(pos_list[1])
    return xs,ys

def centroids2(img_segm,liste):
    """Computes the positions of centroids in a segmented image.
    Returns the centroids numbers in order."""
    m = len(liste)
    xs = np.zeros(m)
    ys = np.zeros(m)
    j=0
    for elt in liste:
        pos_list = np.where(img_segm==elt)
        xs[j] = np.mean(pos_list[0])
        ys[j] = np.mean(pos_list[1])
        j+=1
    return xs,ys

def w_thresh(gray):
    """Wraps up the thresholding method"""
    thresh = sf.canny_threshold(gray)
    kernel = np.ones((4,4), np.uint8)
    thresh = cv2.erode(thresh, kernel, iterations=1)
    return thresh
            

"""niou methods"""
def updateLabels(correspondance_list,labels_list,i,image):
    """updates the list of labels based on the correspondance list
    image is the new image to update"""
    l_corres = len(correspondance_list)
    l_prev_index = len(labels_list[(i-1)%2])
    l_curr_index = len(labels_list[i%2])
    prev_index = labels_list[(i-1)%2]
    index_changes = []
    ref_image = np.copy(image)

    if l_prev_index==l_curr_index:   #Same number of cells
        
        for x,y in correspondance_list: 
            labels_list[i%2][y] = prev_index[x]
            image[ref_image==y+1] = prev_index[x]
            
    if l_curr_index > l_prev_index:  # Apparition
        new_index = max(prev_index)+2
        for x,y in correspondance_list: 
            if x<l_prev_index:
                labels_list[i%2][y] = prev_index[x]                
                image[ref_image==y+1] = prev_index[x]
                
            else:
                labels_list[i%2][y] = new_index
                image[ref_image==y+1] = new_index
                index_changes.append(new_index)
                new_index+=1
       
    if l_curr_index < l_prev_index:  # Disparition
        for x,y in correspondance_list: 
            if y<l_curr_index:
                labels_list[i%2][y] = prev_index[x]
                image[ref_image==y+1] = prev_index[x]
            else:
                index_changes.append(prev_index[x])
    return index_changes

def fillCostMatrix(xs0,ys0,xs1,ys1):
    """Computes a cost matrix for the different distances between centroids
    The rows represent the centroids in the previous frame
    The columns the centroids in the next frame
    Returns the cost matrix"""
    M = int ( max(len(xs0),len(xs1)) ) #Number of centroids.
    costMatrix = np.zeros((M,M))
    x_rows = np.zeros(M)
    x_rows[0:len(xs0)] = xs0
    y_rows = np.zeros(M)
    y_rows[0:len(xs0)] = ys0
    
    x_cols = np.zeros(M)
    x_cols[0:len(xs1)] = xs1
    y_cols = np.zeros(M)
    y_cols[0:len(xs1)] = ys1

    for i in range(len(xs0)):
        for j in range(len(xs1)):
            costMatrix[i,j]=(y_rows[i]-y_cols[j])**2
            costMatrix[i,j] += (x_rows[i]-x_cols[j])**2
    return costMatrix


def w_hungarian(centroids_buffer,i,index_disappeared,images_buffer,labels_list):
    #Hungarian algorithm
    xs0,ys0 = centroids_buffer[(i-1)%2]
    xs1,ys1 = centroids_buffer[i%2]
    size_diff = len(xs1)-len(xs0)  #Is positive if a cell appears and negative if one disappears.

    cost_matrix = fillCostMatrix(xs0,ys0,xs1,ys1)
    hungry_test = hungarian.Hungarian(input_matrix=cost_matrix)
    hungry_test.calculate()
    correspondance_list = hungry_test.get_results()
    image_to_update  = images_buffer[:,:,i%2]   #This labeled image will be re labeled by updateLabels
    disappeared = updateLabels(correspondance_list,labels_list,i,image_to_update)
    if size_diff!=0:
        index_disappeared.append((i,size_diff,disappeared))
    return image_to_update               
                
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
        

def showSegmentation_color(filename,total_buffer,save=False,savename='video8/'):
    """Based on a pre-established segmentation shows the corresponding movie"""
    cap = cv2.VideoCapture(filename)

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
            cv2.imwrite(savename+'frame'+str(i)+'.png',frame)


def filter_out_flickers(total_buffer,index_disappeared):
    """Removes flickering artefacts considering that a signal flickering in time is an artefact.
    Can only handle one flickering thing at a time for now."""
    
    wait_for_disparition = False
    candidate_for_disparition = -1
    to_destroy = []   #List of 3D tuples (value,first_index,last_index) of segmented elements to remove from image
    beginning_index = -1
    premier_i =-1
    list_of_is =[]
    
    previous_index, osef2, osef3 = index_disappeared[0]  #Get the index for the first event

    for i in range(0,len(index_disappeared)):
        index,diff,list_index = index_disappeared[i]
        time_thr = 5   
        #Remove an appearing and disappearing object from the experiment only if it
        #disappears in the next 5 (arbitrary) frames. If longer, conseder that something relevant
        #happened.
        
        if wait_for_disparition:
            #If sth appeared, destroy it if:
            #-It is the same object that disappears
            #-If the event is a disparition
            #-If it disappears in a time<time_thr
            size = np.count_nonzero(total_buffer[:,:,index-1]==list_index[0])
            if list_index[0]==candidate_for_disparition and diff<0 and size<500:
                to_destroy.append((list_index[0],beginning_index,index))
                list_of_is.append(premier_i)
                list_of_is.append(i)
            wait_for_disparition=False
            
        if diff>0:   #Creation, do wait for disparition
            candidate_for_disparition = list_index[0]
            beginning_index = index
            wait_for_disparition =True
            premier_i = i
            
    return to_destroy,list_of_is

def remove_to_destroy(total_buffer,to_destroy):
    """Removes the segmented elements of low interest"""
    totbuf=np.copy(total_buffer)
    for val,begInd,endInd in to_destroy:
        for j in range(endInd-begInd):
            index_beg = begInd+j
            totbuf[ total_buffer[:,:,index_beg]==val,index_beg]=0
    return totbuf

def showSegmentation(filename,total_buffer,save=False,savename='video7/'):
    """Based on a pre-established segmentation shows the corresponding movie"""
    cap = cv2.VideoCapture(filename)

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
            cv2.imwrite(savename+'frame'+str(i)+'.png',total_buffer[:,:,i])
            
            
def get_remaining_events(index_disappeared,to_destroy):
    """From the list of events and the list of elements destroyed, returns
    the list of events remaining to be processed."""
    index_cp = index_disappeared[:]
    for i,deb,fin in to_destroy:
        index_cp = [(x,y,z) for x,y,z in index_cp if (x!=deb and x!=fin)]
    return index_cp

def isFusion(event,buff):
    """Tests is the triplet event (index,difference,label) is a fusion event"""
    index,diff,label = event
    label = label[0]
    if diff>0:
        return False,[]
    img_before = np.copy(buff[:,:,index-1])
    img_after = np.copy(buff[:,:,index])
    mask_before = (img_before==label).astype(np.uint8)
    nb_elts_before = np.amax(img_before)
    kernel = np.ones((7,7),np.uint8)
    neighbouring_mask = cv2.dilate(mask_before,kernel,iterations=8)

    new_map = np.multiply(img_before,neighbouring_mask.astype(np.uint8))
    
    #Removing the element we are currently looking at
    new_map[img_before==label]=0
    possible_candidates = []
    for i in range(nb_elts_before):
        if np.any(new_map==i+1):
            possible_candidates.append(i+1)
    #Computes the area of the cells and compares them
    size_cell_disappearing = np.count_nonzero(img_before==label)
    match = []   #lists the ratios sizeAfter/sizeBefore for possible matches
    
    for vals in possible_candidates:
        size_other_cell = np.count_nonzero(img_before==vals)
        size_before = size_other_cell+size_cell_disappearing
        size_after = np.count_nonzero(img_after==vals)
        ratio = float(size_after)/float(size_before)
        if ratio>0.8 and ratio<1.2:
            match.append((vals,abs(1-ratio)))
    if len(match)==0:
        return False,[]
    if len(match)>1:
        #Several matches, so pick the best
        values = [y for x,y in match]
        result_label,osef = match[np.argmin(values)]
    else:
        result_label, osef = match[0]
    return True,result_label

def isDivision(event,buff):
    """Tests is the triplet event (index,difference,label) is a Division event"""
    index,diff,label = event
    label = label[0]
    if diff<0:
        return False,[]
    img_before = np.copy(buff[:,:,index-1])
    img_after = np.copy(buff[:,:,index])
    mask_after = (img_after==label).astype(np.uint8)
    nb_elts_after = np.amax(img_after)
    kernel = np.ones((7,7),np.uint8)
    neighbouring_mask = cv2.dilate(mask_after,kernel,iterations=8)

    
    new_map = np.multiply(img_after,neighbouring_mask.astype(np.uint8))    
    #Removing the element we are currently looking at
    new_map[img_after==label]=0
    possible_candidates = []
    for i in range(nb_elts_after):
        if np.any(new_map==i+1):
            possible_candidates.append(i+1)
    #Computes the area of the cells and compares them
    size_cell_after = np.count_nonzero(img_after==label)
    match = []   #lists the ratios sizeAfter/sizeBefore for possible matches
    for vals in possible_candidates:
        size_before = np.count_nonzero(img_before==vals)
        size_other_cell = np.count_nonzero(img_after==vals)
        size_after = size_cell_after + size_other_cell
        ratio = float(size_after)/float(size_before)
        if ratio>0.8 and ratio<1.2:
            match.append((vals,abs(1-ratio)))
    if len(match)==0:
        return False,[]
    if len(match)>1:
        #Several matches, so pick the best
        values = [y for x,y in match]
        result_label,osef = match[np.argmin(values)]
    else:
        result_label, osef = match[0]
    return True,result_label


def splitCell(buff,index,ref_label,new_label):
    """Splits a cell into two"""
    cell_before = np.copy(buff[:,:,index-1])
    cell_after = np.copy(buff[:,:,index])
    
    mask_after = cell_after ==ref_label
    
    cell_before[np.logical_not(mask_after)] = 0
    
    mask_ref_label = cell_before ==ref_label
    mask_new_label = cell_before==new_label
  
    after_sure_ref = np.logical_and(mask_ref_label,mask_after)
    after_sure_new = np.logical_and(mask_new_label,mask_after)
    after_unsure = np.logical_and(mask_after,np.logical_not(np.logical_or(after_sure_ref,after_sure_new) ) )

    xref,yref = np.where(after_sure_ref)
    ref_pts = np.concatenate((xref.reshape(-1,1),yref.reshape(-1,1)),axis=1)
    xnew,ynew = np.where(after_sure_new)
    new_pts = np.concatenate((xnew.reshape(-1,1),ynew.reshape(-1,1)),axis=1)
    
    labels_ref = np.ones(xref.shape[0])
    labels_new = np.zeros(xnew.shape[0])
    labels = np.concatenate((labels_ref,labels_new),axis=0)
    labels.reshape(-1,1)
    X= np.concatenate((ref_pts,new_pts),axis = 0)
    
    xu,yu = np.where(after_unsure)
    u_pts = np.concatenate((xu.reshape(-1,1),yu.reshape(-1,1)),axis=1)
    neigh = KNeighborsClassifier(n_neighbors=5)
    neigh.fit(X, labels)
    pred = neigh.predict(u_pts)
    for i in range(pred.shape[0]):
        #if pred is 1 goes to ref if 0 goes to new
        if pred[i]==1:
            after_sure_ref[u_pts[i,0],u_pts[i,1]]=True
        else:
            after_sure_new[u_pts[i,0],u_pts[i,1]]=True
    #Assigning the new values to the thing:
    buff[after_sure_ref,index] = ref_label
    buff[after_sure_new,index] = new_label
    
def bindEvents(fusionEvents,divisionEvents, buff):
    """Relates a fusion to a division event"""
    #1/Finding correspondances
    fusion_indices = []
    fusion_labels = []
    fusion_labels_2 = []  # In label 2 says with which cell the disappearded one has
    for events,label in fusionEvents:
        index,osef,labels = events
        fusion_indices.append(index)
        fusion_labels.append(labels[0])
        fusion_labels_2.append(label)
        
    division_indices = []
    division_labels = []
    division_labels_2 = []  # Tells in which cell it is created
    for events,label in divisionEvents:
        index,osef,labels = events
        division_indices.append(index)
        division_labels.append(labels[0])
        division_labels_2.append(label)
    
    associated_division_list = []
    associated_indexes = []
    associated_labels = []
    for i in fusion_indices:
        ind = next((x for x in division_indices if x>i),-1)
        if ind>0:
            associated_division_list.append((i,ind))
            corr_ind_fusion = fusion_indices.index(i)
            corr_ind_division = division_indices.index(ind)
            associated_indexes.append((corr_ind_fusion,corr_ind_division))
            label1 = fusion_labels
            
    #2/removing corresponding elements
    for j in range(len(associated_division_list)):
        index_fus, index_div = associated_indexes[j]
        if division_labels_2[index_div]==fusion_labels_2[index_fus]:
            #If they are not equal, means that the process of division/fusion 
            #has not happened on the same blob and hence is not relevant
            big_label = division_labels_2[index_div]
            small_label = fusion_labels[index_fus]
            new_label = division_labels[index_div] #Replace after division this label by small label
            first_index = fusion_indices[index_fus]
            second_index = division_indices[index_div]
            
            for k in range(second_index-first_index):
                splitCell(buff,first_index+k,big_label,small_label)
                
            #Attribution of the new created cells to each one of the previous cells:
            #For this, we take the closest centroid
            #centroid of the big label
            last_image = buff[:,:,second_index]
            xs,ys = centroids2(last_image,[big_label,new_label])
            xs0,ys0 = centroids2(buff[:,:,second_index-1],[big_label,small_label])
            dist_regular = (xs0[0]-xs[0])**2 + (ys0[0]-ys[0])**2 + (xs0[1]-xs[1])**2 + (ys0[1]-ys[1])**2
            dist_inverted = (xs0[0]-xs[1])**2 + (ys0[0]-ys[1])**2 + (xs0[1]-xs[0])**2 + (ys0[1]-ys[0])**2
            
            if dist_regular>dist_inverted:
                print "ca marche pas gael euh quoi?"
                tmp_stack = buff[:,:,second_index:]
                tmp_stack[buff[:,:,second_index:]==big_label]=small_label
                tmp_stack[buff[:,:,second_index:]==new_label]=big_label
                buff[:,:,second_index:] = tmp_stack
                division_labels = [x if (x!=new_label and x!=big_label) else big_label if x==new_label else small_label for x in division_labels]
                fusion_labels = [x if x!=new_label and x!=big_label else big_label if x==new_label else small_label for x in fusion_labels]
                division_labels_2= [x if x!=new_label and x!=big_label else big_label if x==new_label else small_label for x in division_labels_2]
                fusion_labels_2= [x if x!=new_label and x!=big_label else big_label if x==new_label else small_label for x in fusion_labels_2]
            else:
                print "ca marche bien gael"
                """Reassigning new labels"""
                tmp_stack = buff[:,:,second_index:]
                tmp_stack[tmp_stack==new_label] = small_label
                buff[:,:,second_index:] = tmp_stack
                division_labels = [x if x!=new_label else small_label for x in division_labels]
                fusion_labels = [x if x!=new_label else small_label for x in fusion_labels]
                division_labels_2 = [x if x!=new_label else small_label for x in division_labels_2]
                fusion_labels_2 = [x if x!=new_label else small_label for x in fusion_labels_2]
                
                
"""1st pass"""

print "1st pass"
filename = 'itoh-cell-migration-02.mov'

cap = cv2.VideoCapture(filename)

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
save=True
filename='video1/'
saveLabels = 0
labels_list=[[0],[0]]
total_buffer= np.zeros((height-30,width,length),dtype=np.uint8)
for i in range(length):
    #Get frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray=gray[30:,:]
    #Definition of the different quantities used for temporal analysis
    if i==0:
        images_buffer = np.zeros((gray.shape[0],gray.shape[1],2))
        gray_buffer = np.zeros((gray.shape[0],gray.shape[1],2),dtype=np.uint8) # remembers i-1 and i-2
        centroids_buffer = [0,0]   #List with two entries
        #total_buffer = np.zeros((gray.shape[0],gray.shape[1],length),dtype=np.uint8)
        index_disappeared = [] 
        #List3d tuples (i,size_diff,indexes) with the iteration number, the difference between number of cells found
        #between two frames and the index of the cell which appeared or disappeared.  
    #thresholding
    thresh = w_thresh(gray)
    gray_buffer[:,:,i%2] = gray
    #Segmentation
    labels_new, nlabs = ndi.label(thresh)
    labels_new_filtered = filter_by_size(labels_new)
    
    #Misc. assigmnments
    images_buffer[:,:,i%2] = labels_new_filtered  
    total_buffer[:,:,i] = images_buffer[:,:,i%2]
    labels_list[i%2] = range(1,np.amax(labels_new_filtered)+1)
    centroids_buffer[i%2] = centroids2(labels_new_filtered,labels_list[i%2])
    
    if i==15:
        saveLabels = np.copy(images_buffer)
    #Hungarian algorithm
    if i>0:
        images_buffer[:,:,i%2] = w_hungarian(centroids_buffer,i,index_disappeared,images_buffer,labels_list)
        total_buffer[:,:,i] = images_buffer[:,:,i%2]
        
    gray_contours = drawContours(gray_buffer[:,:,i%2],images_buffer[:,:,i%2]==3)
    xc,yc = centroids_buffer[i%2]
    drawCentroids(gray_contours,xc,yc)
        
cap.release()
filename = 'itoh-cell-migration-02.mov'
showSegmentation_color(filename,total_buffer,save=True,savename='video1/')

"""2nd pass"""
print "2nd pass"
tota_buf = np.copy(total_buffer)
to_d,list_of_is = filter_out_flickers(tota_buf,index_disappeared)
buff = remove_to_destroy(tota_buf,to_d)
filename = 'itoh-cell-migration-02.mov'

rem_events = get_remaining_events(index_disappeared,to_d)

"""Defuse the fused people"""
fusionEvents=[]
divisionEvents = []
for elts in rem_events:
    boole,label = isFusion(elts,total_buffer)
    boole2,label2 = isDivision(elts,total_buffer)
    if boole:
        fusionEvents.append((elts,label))
    if boole2:
        divisionEvents.append((elts,label2))

buffe = np.copy(buff)
bindEvents(fusionEvents,divisionEvents,buffe)

#showSegmentation_color(filename,buffe,save=True,savename='video_color/')

fusionEvents=[]
divisionEvents = []
for elts in rem_events:
    boole,label = isFusion(elts,total_buffer)
    boole2,label2 = isDivision(elts,total_buffer)
    if boole:
        fusionEvents.append((elts,label))
    if boole2:
        divisionEvents.append((elts,label2))

buffe = np.copy(buff)
bindEvents(fusionEvents,divisionEvents,buffe)

showSegmentation_color(filename,buffe,save=True,savename='video2/')