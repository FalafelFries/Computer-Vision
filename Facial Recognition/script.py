#Facial Recognition

#importing required libraries
import cv2, os, numpy as np

path1 = os.getcwd() + r"\Facial Recognition" #getting current working directory
alg = path1 + r"\haarcascade_frontalface_default.xml" #storing path of haarcascadeface algorithm

haar_cascade = cv2.CascadeClassifier(alg) #loading the haarcascadeface algorithm

dataset = input("Enter name of dataset folder :") #getting name of the dataset folder

#reading data from dataset...

print("Reading data from dataset...") 

#initialising images, labels, a names and id
#images are stored as NumPy arrays
#each person in the dataset has a label associated with it. label is basically an integer representing each person
#names contains names of the people in the dataset
#id is basically used as a counter to access names within a dataset

(images, labels, names, id) = ([], [], {}, 0)

#filling up images, labels and names
#dir is the dataset folder
#subdirectories of dataset are names
#files are the images present in names

for(subdirs, dir, file) in os.walk(os.path.join(path1, dataset)) :
    
    for subdirs in dir : #walking into directory where we have subdirectories..

        names[id] = subdirs #storing name
        subjectPath = os.path.join(path1, dataset, subdirs) #storing path of subdirectory

        for file in os.listdir(subjectPath) : #walking into each subdirectory where we have images of people..

            path2 = subjectPath + "/" + file #storing path of image
            label = id #storing label
            images.append(cv2.imread(path2, 0)) #appending image to images
            labels.append(int(label)) #appending label to labels
        id += 1 #incrementing id so as to move onto the next person in the dataset

(images, labels) = [np.array(lis) for lis in [images, labels]] #converting images and labels to NumPy arrays

(width, height) = (500, 500) #storing width and height of images

model = cv2.face.LBPHFaceRecognizer_create() #creating LBPH face recognizer

#training the data...

print("Training the data...")

model.train(images, labels) 

print("Training complete!")

cam = cv2.VideoCapture(0) #initialising camera

while True : #infinitely loops over the frames of the video

    _, img = cam.read() #reading from camera
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #greyscaling image

    face = haar_cascade.detectMultiScale(grayImg, 1.3, 5) #getting face coordinates

    for(x, y, w, h) in face :
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)  #drawing bounding box around face on webcam
        faceOnly = grayImg[y:y+h, x:x+w] #cropping out face on webcam
        resizeImg = cv2.resize(faceOnly, (width, height)) #resizing image on webcam

        #predicting whether the image on webcam belongs to the dataset or not

        prediction = model.predict(resizeImg)

        #prediction[0] gives name of the person prediction[1] gives confidence of prediction

        if prediction[1] < 40 : #image on webcam is present in dataset

            cv2.putText(img, "%s - %.0f" %(names[prediction[0]], prediction[1]), (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255)) #displaying text
            print(names[prediction[0]]) #printing name

        else : #image on webcam is not present in dataset

            cv2.putText(img, "Unknown", (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0)) #displaying text
            print("Unknown Person") #printing name

    cv2.imshow("Face Detection", img) #displaying frame

    key1 = cv2.waitKey(10) & 0xFF #waitKey() waits for a pressed key
    if key1 == ord("q") : #if the `q` key is pressed, break from the loop
        break

cam.release() #releasing captured object
cv2.destroyAllWindows() #closing any open windows