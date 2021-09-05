#Face Detection using webcam

#importing required libraries

import cv2
import os

dir = os.getcwd() #getting current working directory
path = dir + r"\from webcam\haarcascade_frontalface_default.xml" #storing path of haarcascadeface algorithm

haar_cascade = cv2.CascadeClassifier(path) #loading the haarcascadeface algorithm

cam = cv2.VideoCapture(0) #initialising camera

while True : #infinitely loops over the frames of the video

    _, img = cam.read() #reading from camera
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscaling image

    face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) #getting face coordinates
    
    for(x, y, w, h) in face : #drawing bounding box around face
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow("Face Detection", img) #displaying frame

    key = cv2.waitKey(10) & 0xFF #waitKey() waits for a pressed key
    if key == ord("q") : #if the `q` key is pressed, break from the loop
        break

cam.release() #releasing captured object
cv2.destroyAllWindows() #closing any open windows