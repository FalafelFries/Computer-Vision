#Finding HSV of object to be tracked

#importing required packages
import cv2
import numpy as np

cam = cv2.VideoCapture(0) #initialising camera

cv2.namedWindow("HSV Value") #creating a window named 'HSV Value'

#'optional' argument is required for trackbar creation parameters
def nothing(x) : 
    pass

#creating trackbars
cv2.createTrackbar("H High", "HSV Value", 0, 179, nothing)
cv2.createTrackbar("H Low", "HSV Value", 0, 179, nothing)
cv2.createTrackbar("S High", "HSV Value", 0, 255, nothing)
cv2.createTrackbar("S Low", "HSV Value", 0, 255, nothing)
cv2.createTrackbar("V High", "HSV Value", 0, 255, nothing)
cv2.createTrackbar("V Low", "HSV Value", 0, 255, nothing)

while True : #infinitely loops over the frames of the video

    _, img = cam.read() #reading from camera
    img = cv2.GaussianBlur(img, (5, 5), 0) #smoothening image
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #converting to HSV

    #reading hsv values from trackbar
    h_high = cv2.getTrackbarPos("H High", "HSV Value")
    h_low = cv2.getTrackbarPos("H Low", "HSV Value")
    s_high = cv2.getTrackbarPos("S High", "HSV Value")
    s_low = cv2.getTrackbarPos("S Low", "HSV Value")
    v_high = cv2.getTrackbarPos("V High", "HSV Value")
    v_low = cv2.getTrackbarPos("V Low", "HSV Value")

    #we need our final frame to display only the object which contains the specefied HSV values

    #declaring an array that would store the final hsv values
    hsv_low = np.array([h_low, s_low, v_low])
    hsv_high = np.array([h_high, s_high, v_high])

    #maskk contains the HSV values of the object
    maskk = cv2.inRange(hsv, hsv_low, hsv_high)

    #cv2.bitwise_and() allows you to display your maskk back onto your initial frame
    res = cv2.bitwise_and(img, img, mask = maskk)

    cv2.imshow("HSV Value", img) #displaying your camera
    cv2.imshow("Mask", maskk) #displaying maskk
    cv2.imshow("Result", res) #displaying result

    key = cv2.waitKey(10) & 0xFF #waitKey() waits for a pressed key
    if key == ord("q") : #if the `q` key is pressed, break from the loop
        break

cam.release() #releasing captured object
cv2.destroyAllWindows() #closing any open windows