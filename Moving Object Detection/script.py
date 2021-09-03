#importing required libraries

import cv2
import imutils
import time

cam = cv2.VideoCapture(0) #initialising camera
time.sleep(1) #delaying execution for 1s

firstFrame = None #initialising the first frame
area = 500

while True : #infinitely loops over the frames of the video

    _, img = cam.read() #reading from camera
    text = "normal" #ie, no motion is taking place
    img = imutils.resize(img, width = 500) #resizing image
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscaling image
    gaussianImg = cv2.GaussianBlur(grayImg, (21, 21), 0) #smoothening image

    if firstFrame is None : #initialising first frame if not initialised 
        firstFrame = gaussianImg 
        continue

    imgDiff = cv2.absdiff(firstFrame, gaussianImg) #computing the absolute difference between the current frame and the first frame

    threshImg = cv2.threshold(imgDiff, 25, 255, cv2.THRESH_BINARY)[1] #applying threshold
    threshImg = cv2.dilate(threshImg, None, iterations=2) #removing noise from image

    cnts = cv2.findContours(threshImg.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #finding contours
    cnts = imutils.grab_contours(cnts)

    for c in cnts : #looping over the contours
        if cv2.contourArea(c) < area : #if the contour is too small, ignore it
            continue
        (x, y, w, h) = cv2.boundingRect(c) #computing bounding box for contour
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #displaying bounding box
        text = "moving object detected" #updating text
    print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2) #displaying text
    cv2.imshow("cameraFeed", img) #displaying frame
    key = cv2.waitKey(1) & 0xFF #waitKey() waits for a pressed key
    if key == ord("q") : #if the `q` key is pressed, break from the loop
        break

cam.release() #releasing captured object
cv2.destroyAllWindows #closing any open windows