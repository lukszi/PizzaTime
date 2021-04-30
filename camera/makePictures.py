import time
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera

################################
wCam, hCam = 320, 240
################################

# cap = cv2.VideoCapture(0)

camera = PiCamera()
camera.resolution = (1280, 720)
camera.framerate = 10
rawCapture = PiRGBArray(camera, size=(1280, 720))

time.sleep(0.1)

# cap.set(3, wCam)
# cap.set(4, hCam)
iterator = 0
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    # grab the raw NumPy array representing the image, then initialize the timestamp
    # and occupied/unoccupied text
    img = frame.array
    print(img.shape)
    filename = f'data/{str(iterator)}.jpg'
    iterator += 1
    cv2.imwrite(filename, img)

    rawCapture.truncate(0)

    time.sleep(1)
