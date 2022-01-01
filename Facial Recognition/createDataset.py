#Creating Dataset for Facial Recognition

#importing required libraries

import cv2
import os

dir1 = os.getcwd() #getting current working directory

#appending folder name
folder = r"\Facial Recognition"
dir2 = dir1 + folder

dataset = input("Enter name of dataset : ") #storing name of the folder containing the dataset

while True : #infinite loop for storing datasets of different people

    #to terminate from this loop after creating all datasets, press Ctrl+Z

    name = input("Enter name of person : ") #storing name of person

    path1 = os.path.join(dir2, dataset, name) #makes the path ../Facial Recognition/<dataset>/<name>
    if not os.path.exists(path1) : #makes directory if it doesn't exist
        os.makedirs(path1)

    (width, height) = (500, 500) #storing width and height of images

    path2 = dir2 + r"\haarcascade_frontalface_default.xml" #storing path of haarcascadeface algorithm
    haar_cascade = cv2.CascadeClassifier(path2) #loading the haarcascadeface algorithm

    cam = cv2.VideoCapture(0) #initialising camera

    count = 1 #initialising count

    while count <= 100 : #looping over 100 times so as to capture 100 images of person

        print(count) #printing count
        _, img = cam.read() #reading from camera
        grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscaling image

        face = haar_cascade.detectMultiScale(grayImg, 1.3, 4) #getting face coordinates

        for(x, y, w, h) in face : 
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2) #drawing bounding box around face
            faceOnly = grayImg[y:y+h, x:x+w] #cropping out face
            resizeImg = cv2.resize(faceOnly, (width, height)) #resizing image
            cv2.imwrite("%s/%s.jpg" %(path1, count), resizeImg) #saving image in <dataset>/<name>/m for mth image
            count+=1 #incrementing count

        cv2.imshow("Face Detection", img) #displaying frame

        key1 = cv2.waitKey(10) & 0xFF #waitKey() waits for a pressed key
        if key1 == ord("q") : #if the `q` key is pressed, break from the loop
            break

    print("Images captured successfully")

    cam.release() #releasing captured object
    cv2.destroyAllWindows() #closing any open windows