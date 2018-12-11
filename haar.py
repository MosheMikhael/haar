# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 13:41:09 2018

@author: moshe.f
"""
import os
import numpy as np
import cv2 as cv
import time
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
#eye_cascade = cv.CascadeClassifier('haarcascade_eye.xml')
#img = cv.imread('sachin.jpg')
cam = cv.VideoCapture(0)
#cam = cv.VideoCapture('IMG_8204.mp4')
#files = os.listdir('Cars_driver_side')
#for path in files:
while True:
#    img = cv.imread('Cars_driver_side/' + path)

    _, img = cam.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    T = time.clock()
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    print(time.clock() - T)
    for (x,y,w,h) in faces:
        cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
    #    eyes = eye_cascade.detectMultiScale(roi_gray)
    #    for (ex,ey,ew,eh) in eyes:
    #        cv.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    img = cv.resize(img, (800, 600))
    cv.imshow('img',img)
    
    k = cv.waitKey(1) & 0xff
    if k == 27:
        break
    
cv.destroyAllWindows()
cam.release()