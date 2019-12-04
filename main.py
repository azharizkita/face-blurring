import cv2
import os
import numpy as np

class prop(object):
    def __init__(self, convolutionKernelDimension = 3, faceCascade = "haarcascade_frontalface_default.xml"):
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + faceCascade)
        self.convolutionKernel = np.ones((convolutionKernelDimension, convolutionKernelDimension)) * (1 / (convolutionKernelDimension**2)) 
        self.mainFilePath = os.path.dirname(__file__)

class displayWindow(object):
    def __init__(self, displayName = 'img', resolution = (600, 600)):
        self.displayName = displayName
        self.displayWindowName = cv2.namedWindow(displayName, cv2.WINDOW_NORMAL)
        self.displayWindowSize = cv2.resizeWindow(displayName, resolution[0], resolution[1])

    def showImage(self, image):
        cv2.imshow(self.displayName, image)

if __name__ == "__main__":
    properties = prop(convolutionKernelDimension = 21)
    original = displayWindow('Original', (650, 650))
    censored = displayWindow('Censored', (650, 650))

    for target_list in os.listdir(properties.mainFilePath+'/Wed5a'):
        print(target_list)
        img = cv2.imread(properties.mainFilePath+'/Wed5a/'+target_list)
        
        original.showImage(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = properties.faceCascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            face = cv2.filter2D(
                src=img[y:y+h, x:x+w], kernel=properties.convolutionKernel, ddepth=-1)
            img[y:y+h, x:x+w] = face
        cv2.imshow('Censored', img)
        censored.showImage(img)
        cv2.waitKey()

    print('Done\n')
