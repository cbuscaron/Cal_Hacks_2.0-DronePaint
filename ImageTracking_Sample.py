# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 22:22:49 2015

@author: Camilo
"""

import cv2
import cv2.cv as cv
import numpy as np


cap = cv2.VideoCapture(1)

cap.set(cv.CV_CAP_PROP_FRAME_WIDTH, 640);
cap.set(cv.CV_CAP_PROP_FRAME_HEIGHT, 480); 

while(1):    
    ret, frame = cap.read()
    #frame2 = frame.copy()
    #frame = cv2.blur(frame,(5,5))
    kernel = np.ones((3,3),np.uint8)
    
    #RGB detection part,it is not much possible to track specific colored objects using RGB
    b,g,r = cv2.split(frame)
    (thresh, bw) = cv2.threshold(b, 128, 255, cv2.THRESH_BINARY)
    bw = cv2.erode(bw,kernel,iterations = 1)
    
    
    #bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)
    bw=cv2.dilate(bw,kernel,iterations = 5)
    bw = cv2.blur(bw,(10,10))
    #HSV detection and tracking
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([110,50,50], dtype=np.uint8)
    upper = np.array([130,255,255], dtype=np.uint8)
    
    #best_cnt=np.array([0,0], dtype=np.uint8)
    
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.erode(mask,kernel,iterations = 2)
    #mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.blur(mask,(5,5))
    mask2 = mask.copy()
    cv2.imshow('HSV object detection',mask2)
    contours,hierarchy = cv2.findContours(mask,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
    #print(best_cnt)  
    M = cv2.moments(best_cnt)
    cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
    cv2.circle(frame,(cx,cy),10,(10,0,255),-1)
    res = cv2.bitwise_and(frame,frame, mask= mask)
    cv2.imshow('HSV based object tracking',frame)
    cv2.imshow('RGB based detection ',bw)
    #cv2.imshow('HSV Object Detection',mask2)
    #cv2.imshow('Original Video',frame2)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()