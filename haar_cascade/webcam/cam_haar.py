import cv2
import numpy as np

cap = cv2.VideoCapture(0)

faceClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeClassif = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye_tree_eyeglasses.xml")

while True:
    ret,frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceClassif.detectMultiScale(gray, 1.3, 5)
    eyes = eyeClassif.detectMultiScale(gray, 1.3, 5)

    #draw_faces(faces, frame, (0,255,255))
    #draw_faces(eyes, frame, (0,0,255))

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),2)
    for (x,y,w,h) in eyes:
        cv2.rectangle(frame, (x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('Webcam face detection',frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

#https://github.com/opencv/opencv/tree/master/data/haarcascades  -- modelos preentrenados
#https://www.youtube.com/watch?v=J1jlm-I1cTs&t=2s
#https://www.youtube.com/watch?v=7V-228jPG4s