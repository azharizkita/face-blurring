import cv2, os
__dir__ = os.path.dirname(__file__)
face_cascade = cv2.CascadeClassifier(__dir__+'/haarcascade_frontalface_default.xml')

for target_list in os.listdir(__dir__+'/Wed5a'):
    print(target_list)
    img = cv2.imread(__dir__+'/Wed5a/'+target_list)
    cv2.imshow('img', img)
    cv2.waitKey()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('img', gray)
    cv2.waitKey()
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow('img', img)
    cv2.waitKey()