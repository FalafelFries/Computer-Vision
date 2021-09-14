#Object Tracking based on Colour

#importing required packages
import cv2
import imutils

print("Enter HSV values :")
h_low = input("H low = ")
h_high = input("H high = ")
s_low = input("S low = ")
s_high = input("S high = ")
v_low = input("V low = ")
v_high = input("V high = ")

obj_low = (int(h_low), int(s_low), int(v_low)) #HSV low values of object to be tracked
obj_high = (int(h_high), int(s_high), int(v_high)) #HSV high values of object to be tracked

cam = cv2.VideoCapture(0) #initialising camera

while True : #infinitely loops over the frames of the video

    (grabbed, img) = cam.read() #reading from camera

    img = imutils.resize(img, width = 600) #resizing image
    blurImg = cv2.GaussianBlur(img, (11, 11), 0) #smoothening image
    hsv = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV) #converting to HSV

    mask = cv2.inRange(hsv, obj_low, obj_high) #mask contains the HSV values of the object
    #removing noise
    mask = cv2.erode(mask, None, iterations = 2) 
    mask = cv2.dilate(mask, None, iterations = 2)

    count = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2] #finding contours
    center = None #initialising centre of enclosing circle of object to None

    if len(count) > 0 :
        c = max(count, key = cv2.contourArea) #finding maximum area of object so as to form enclosing circle
        ((x, y), radius) = cv2.minEnclosingCircle(c) #forming minimum enclosing circle
        M = cv2.moments(c) #finding moments so as to find centre of enclosing centre
        center = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) #using formula to compute center

        if radius > 10 : #forms enclosing circle only if radius is greater than 10
            cv2.circle(img, (int(x), int(y)), int(radius), (0, 255, 255), 2) #draws circle
            cv2.circle(img, center, 5, (0, 0, 255), -1) #draws a point denoting center of circle
            print(center, radius) #printing center and radius

    cv2.imshow("Object Tracking", img) #displaying frame
    key = cv2.waitKey(1) & 0xFF #waitKey() waits for a pressed key
    if key == ord("q") : #if the `q` key is pressed, break from the loop
        break

cam.release() #releasing captured object
cv2.destroyAllWindows() #closing any open windows