import cv2 as cv
import cv2
import numpy as np 
import os

print(cv.__version__)

recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read('./trainer/trainer.yml')
cascade_path = "./cascades/haarcascade_frontalface_default.xml"
face_cascade = cv.CascadeClassifier(cascade_path)
font = cv2.FONT_HERSHEY_SIMPLEX

id = 0

names = ['None', 'Keith', 'Kaylin']

cap = cv.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

minW = 0.1 * cap.get(3)
minH = 0.1 * cap.get(4)

if not cap.isOpened():
    print("Cannot open source camera")
    exit()


while True:
    ret, frame = cap.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.2, 
        minNeighbors=5, 
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if (100 - confidence) > 50:
            id = names[id]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            id = "Unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv.putText(
            frame,
            str(id),
            (x+5, y-5),
            font,
            1,
            (255,255,255),
            2
        )
        cv.putText(
            frame,
            str(confidence),
            (x+5, y+h-5),
            font,
            1,
            (255,255,0),
            1
        )

    cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
