import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open source camera")
    exit()

while True:
    _, frame = cap.read()

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    lower_green = np.array([36, 25, 25])
    upper_green = np.array([86, 255, 255])
    green_mask = cv.inRange(hsv, lower_green, upper_green)


    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv.inRange(hsv, lower_blue, upper_blue)

    mask = cv.bitwise_or(blue_mask, green_mask)

    res = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)

    # k = cv.waitKey(5) & 0xFF
    # if k == 27:
    #     break
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()