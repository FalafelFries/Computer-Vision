#Facial Emotion Recognition

#importing required libraries

import cv2
from facial_emotion_recognition import EmotionRecognition

''' 
before proceeding further, navigate to ..\Python\Python39\Lib\site-packages\torch\serialization.py and change
def load(f, map_location='gpu', pickle_module=pickle, **pickle_load_args)
to
def load(f, map_location='cpu', pickle_module=pickle, **pickle_load_args)
'''

er = EmotionRecognition(device = "cpu") #creating emotion recognizer

cam = cv2.VideoCapture(0) #initialising camera

while True : #infinitely loops over the frames of the video

    _, img = cam.read() #reading from camera
    img = er.recognise_emotion(img, return_type = "BGR") #recognizing emotion

    cv2.imshow("Facial Emotion Recognition", img) #displaying frame

    key = cv2.waitKey(10) & 0xFF #waitKey() waits for a pressed key
    if key == ord("q") : #if the `q` key is pressed, break from the loop
        break

cam.release() #releasing captured object
cv2.destroyAllWindows() #closing any open windows