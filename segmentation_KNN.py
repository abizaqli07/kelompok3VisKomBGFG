import cv2 as cv
import numpy as np

#Load video capture from file 'data/mouse.mp4'
capture = cv.VideoCapture('data/mouse.mp4')

if not capture.isOpened():
    exit(0)

#Create a KNN background subtractor
subsKNN = cv.createBackgroundSubtractorKNN()

while capture.isOpened():

    #Read a frame from the video capture
    re, frame = capture.read()

    if isinstance(frame, type(None)):
        break

    #Resize the frame to 50% of its original size
    scale = 50
    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)

    dim = (width, height)
    image = cv.resize(frame,dim, cv.INTER_AREA)

    #Apply a Gaussian filter to the resized image
    gaussian = np.array([
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0]
    ])/273
    image = cv.filter2D(image,-1,gaussian)

    #Apply the KNN background subtractor to the filtered image
    blobKNN = subsKNN.apply(image)

    #Display the resulting image
    cv.imshow("image knn",blobKNN)

    #Wait for a key press (30ms) and exit if 'q' or ESC is pressed
    keyword = cv.waitKey(30)
    if keyword=='q' or keyword==27:
        break

#Release all resources and exit
cv.destroyAllWindows()
exit(0)