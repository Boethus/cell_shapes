# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 13:39:15 2017

@author: univ4208
"""

import numpy as np
import cv2
import os
filename = os.path.join("data",'itoh-cell-migration-02.mov')
cap = cv2.VideoCapture(filename)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

for i in range(length):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = frame[:,:,0]
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(150) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()