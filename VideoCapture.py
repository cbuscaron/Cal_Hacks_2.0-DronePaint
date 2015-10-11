# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:31:31 2015

@author: Camilo
"""

import numpy as np
import cv2
import cv2.cv as cv


cap = cv2.VideoCapture(1)

cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480); 

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()