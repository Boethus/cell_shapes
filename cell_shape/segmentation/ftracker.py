# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:55:30 2017

@author: univ4208
"""

import numpy as np
import cv2
import scipy.ndimage as ndi
import skimage
import hungarian
import skimage.feature
from sklearn.neighbors import KNeighborsClassifier
a=10

def filter_by_size(img_segm):
    """filters a segmented image by getting rid of the components with too few pixels"""
    
    numbers = np.zeros(np.max(img_segm-1))
    for i in range(1,np.max(img_segm)):
        numbers[i-1] = np.sum(img_segm==i)
        
    indexes = np.arange(1,np.max(img_segm))
    #indexes = indexes[numbers>np.mean(numbers)] #Deletes the 1-pixel elements
    indexes = indexes[numbers>500] #Deletes the 1-pixel elements
    
    segm_filtered = np.zeros(img_segm.shape)
    j=1
    for i in (indexes):
        segm_filtered[img_segm==i] = j
        j+=1
    return segm_filtered
            
def centroids(img_segm):
    """Computes the positions of centroids in a segmented image.
    Returns the centroids numbers in order"""
    m = int(np.amax(img_segm))
    xs = np.zeros(m)
    ys = np.zeros(m)
    
    for i in range(0,m):
        pos_list = np.where(img_segm==i+1)
        xs[i] = np.mean(pos_list[0])
        ys[i] = np.mean(pos_list[1])
    return xs,ys

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

def w_hungarian(prev_image,next_image,max_distance=50):
    """Using the hungarian algorithm, looks for corresponding cells between 
    prev_image and next_image. If a distance is greater than max_distance, it is considered
    as two different cells.
    When a cell is associated with -1, it means that it is not associated with anything yet"""
    xs0,ys0 = centroids(prev_image)
    xs1,ys1 = centroids(next_image)

    cost_matrix = fillCostMatrix(xs0,ys0,xs1,ys1)
    #make it disatvantageous to select any distance >max_distance
    cost_matrix[cost_matrix>(max_distance**2)]=np.max(cost_matrix)
    cost_matrix[cost_matrix==0]=np.max(cost_matrix)
    
    hungry_test = hungarian.Hungarian(input_matrix=cost_matrix)
    hungry_test.calculate()
    correspondance_list = hungry_test.get_results()
    sum_normal=0
    sum_inverted=0
    for coords in correspondance_list:
        sum_normal+=cost_matrix[coords]
        sum_inverted+=cost_matrix[coords[1],coords[0]]
    apparition_list = [] 
    for i, coords in enumerate(correspondance_list):
        if cost_matrix[coords]>max_distance**2:
            correspondance_list[i] = (coords[0],-1)
            apparition_list.append((-1,coords[1]))
    correspondance_list.extend(apparition_list)
    return correspondance_list

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
    for i in fusion_indices:
        ind = next((x for x in division_indices if x>i),-1)
        if ind>0:
            associated_division_list.append((i,ind))
            corr_ind_fusion = fusion_indices.index(i)
            corr_ind_division = division_indices.index(ind)
            associated_indexes.append((corr_ind_fusion,corr_ind_division))

            
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
     