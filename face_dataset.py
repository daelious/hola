import cv2 as cv
import os

cam = cv.VideoCapture(0)
cam.set(3, 640)
cam.set(4, 480)

face_detector = cv.CascadeClassifier('./cascades/haarcascade_frontalface_default.xml')

face_id = input('\nEnter User ID and press <Enter> ==> ')
print("\n Initializing face capture. Look at the camera and wait ...")

count = 0
while True:
    ret, img = cam.read()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        count += 1

        cv.imwrite("dataset/User."+str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv.imshow('image', img)
    k = cv.waitKey(100) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program and Cleaning up.")
cam.release()
cv.destroyAllWindows()