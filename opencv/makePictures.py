import time

import cv2

################################
wCam, hCam = 320, 240
################################

cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
iterator = 0
while True:
    success, img = cap.read()
    filename = str(iterator) + '.jpg'
    iterator += 1
    cv2.imwrite(filename, img)
    time.sleep(1)
