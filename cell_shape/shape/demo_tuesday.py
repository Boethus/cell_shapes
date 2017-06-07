#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:49:01 2017

@author: aurelien
"""

import os
import find_arms as fa
import cPickle as pickle
path2 = os.path.join("..",'data','microglia','
                     
                     
                     _denoised')
path_centers2 = os.path.join("..",'data','microglia','1_centers') 
path_arms2 = os.path.join("..",'data','microglia','1_arms')    


#redefine_labels(path_centers)

experiment2 = fa.Experiment(path2,path_centers2,path_arms2)

experiment2.load()

path = os.path.join("..",'data','microglia','8_denoised')
path_centers = os.path.join("..",'data','microglia','8_centers') 
path_arms = os.path.join("..",'data','microglia','8_arms') 

experiment3 = fa.Experiment(path,path_centers,path_arms)
experiment3.load()

#print experiment3.path
print experiment2.path
classification_exp2=0   #24 elements
classification_exp3=0   #16 elements
with open('classification_results.pkl','rb') as out:
    classification_exp3 = pickle.load(out)

with open('exp2_classification_results.pkl','rb') as out:
    classification_exp2 = pickle.load(out)

"""
import process_trajectories as pt
#Caracs experiment2
pt.plot_multiple_caracs(experiment2,classification_exp2[0:12],4,3,"distance")
pt.plot_multiple_caracs(experiment2,classification_exp2[12:],4,3,"distance")

pt.plot_multiple_caracs(experiment2,classification_exp2[0:12],4,3,"speed")
pt.plot_multiple_caracs(experiment2,classification_exp2[12:],4,3,"speed")

#Carac experiment3
pt.plot_multiple_caracs(experiment3,classification_exp3,4,4,"distance")

pt.plot_multiple_caracs(experiment3,classification_exp3,4,4,"speed")
"""

#---------Show some examples-----------------------------

interesting_exp3 = [-2,-1,4]
fa.show_complex_trajectory(classification_exp3[-2][1],experiment3,00)

interesting_exp2 = [2,4,-2] #Nb2 is super
fa.show_complex_trajectory(classification_exp2[2][1],experiment2,0)




