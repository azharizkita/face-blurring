import cv2
import os
import numpy as np

class prop(object):
    def __init__(self, convolutionKernelDimension = 3, faceCascade = "haarcascade_frontalface_default.xml", eyeCascade = "haarcascade_eye.xml"):
        self.mainFilePath = os.getcwd()
        self.faceCascade = cv2.CascadeClassifier(self.mainFilePath +'/cascades/'+ faceCascade)
        self.eyeCascade = cv2.CascadeClassifier(self.mainFilePath +'/cascades/'+ eyeCascade)
        self.convolutionKernel = np.ones((convolutionKernelDimension, convolutionKernelDimension)) * (1 / (convolutionKernelDimension**2)) 

class displayWindow(object):
    def __init__(self, displayName = 'img', resolution = (600, 600)):
        self.displayName = displayName
        self.displayWindowName = cv2.namedWindow(displayName, cv2.WINDOW_NORMAL)
        self.displayWindowSize = cv2.resizeWindow(displayName, resolution[0], resolution[1])

    def showImage(self, image):
        cv2.imshow(self.displayName, image)
    
    @staticmethod
    def waitToExit():
        cv2.waitKey()

def faceDetect(parameter_list):
    img = cv2.imread(properties.mainFilePath + '/Wed5a/' + parameter_list)
    gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    return properties.faceCascade.detectMultiScale(gray), img, gray

def faceBlur(img, gray, x, y, w, h, eyeCascade=False):
    if eyeCascade == False:
        face = cv2.filter2D(
            src=img[y:y+h, x:x+w], kernel=properties.convolutionKernel, ddepth=-1)
        img[y:y+h, x:x+w] = face
    elif eyeCascade == True:
        eyes = properties.eyeCascade.detectMultiScale(gray[y:y+h, x:x+w])
        if len(eyes) == 2:
            face = cv2.filter2D(
                src=img[y:y+h, x:x+w], kernel=properties.convolutionKernel, ddepth=-1)
            img[y:y+h, x:x+w] = face

if __name__ == "__main__":
    try:
        kernelDimension = int(input('Kernel Dimension (odd integers): '))
        eyeCascadeOption = str(input('Use eye cascade (y/n, default is "no".)? '))
        if eyeCascadeOption == 'n' or eyeCascadeOption == 'N' or eyeCascadeOption == '':
            isEyeCascade = False
        else:
            isEyeCascade = True
        properties = prop(convolutionKernelDimension = kernelDimension)
        original = displayWindow('Original', (650, 650))
        censored = displayWindow('Censored', (650, 650))
        for target_list in os.listdir(properties.mainFilePath+'/Wed5a'):
            print(target_list)
            faces, img, gray = faceDetect(target_list)
            original.showImage(img)
            for (x, y, w, h) in faces:
                faceBlur(img, gray, x, y, w, h, eyeCascade=isEyeCascade)
            censored.showImage(img)
            displayWindow.waitToExit()
        print('Done\n')
    except Exception as errors:
        print(errors)
    except KeyboardInterrupt:
        print('Terminated with Keyboard Interupt.')
    except SystemExit:
        print('Terminated.')
