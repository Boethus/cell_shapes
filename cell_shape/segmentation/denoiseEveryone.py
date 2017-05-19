# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:26:39 2017

@author: univ4208
"""

import methods as m
import os
path = os.path.join("..","data","microglia")
to_process = ['Beacon-2 unst','Beacon-7 LPS','Beacon-8 LPS']

for mov in to_process:
    path_to_denoise = os.path.join(path,mov)
    path_to_save = os.path.join(path,mov[7]+"_denoised")
    m.denoiseStack(path_to_denoise,path_to_save)