#Face Detection from image

#importing required libraries

import cv2
import os

dir = os.getcwd() #getting current working directory
print(dir)
path1 = dir + r"\Face Detection\from image\myFit.jpg" #storing path of image
img = cv2.imread(path1) #reading image
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscaling image

path2 = dir + r"\Face Detection\from image\haarcascade_frontalface_default.xml" #storing path of haarcascadeface algorithm
haar_cascade = cv2.CascadeClassifier(path2) #loading the haarcascadeface algorithm

face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) #getting face coordinates

for(x, y, w, h) in face : #drawing bounding box around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imwrite("Face Detection.jpg", img) #saving image