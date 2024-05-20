import cv2 as cv
import numpy as np

capture = cv.VideoCapture('data/mouse.mp4')
#capture = cv.VideoCapture(0)
if not capture.isOpened():
    exit(0)
# subsMog2 = cv.createBackgroundSubtractorMOG2(600,125,False)
subsMog2 = cv.createBackgroundSubtractorMOG2()
subsKNN = cv.createBackgroundSubtractorKNN()
i = 0
while capture.isOpened():
    re, frame = capture.read()
    scale = 20

    if isinstance(frame, type(None)):
        break

    width = int(frame.shape[1] * scale / 100)
    height = int(frame.shape[0] * scale / 100)

    dim = (width, height)
    image = cv.resize(frame,dim, cv.INTER_AREA)
    gaussian = np.array([
        [1.0, 4.0, 7.0, 4.0, 1.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [7.0, 26.0, 41.0, 26.0, 7.0],
        [4.0, 16.0, 26.0, 16.0, 4.0],
        [1.0, 4.0, 7.0, 4.0, 1.0]
    ])/273
    image = cv.filter2D(image,-1,gaussian)
    if i == 0:
        frame1 = image
        grayscaleframe1 = cv.cvtColor(frame1, cv.COLOR_RGBA2GRAY)
        i = i+1

    grayscaleframe = cv.cvtColor(image, cv.COLOR_RGBA2GRAY)
    framedelta = cv.absdiff(grayscaleframe1,grayscaleframe)
    _, bgs = cv.threshold(framedelta, 50, 255, cv.THRESH_BINARY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7,7))
    morphImage = bgs
    for i in range(1,3):
        morphImage = cv.morphologyEx(morphImage, cv.MORPH_CLOSE, kernel)
    morphImage = cv.morphologyEx(morphImage, cv.MORPH_DILATE, kernel)

    cv.imshow("image asli",image)
    cv.imshow("image abs diff", bgs)
    cv.imshow("morph image", morphImage)
    keyword = cv.waitKey(30)
    if keyword=='q' or keyword==27:
        break
cv.destroyAllWindows()
exit(0)