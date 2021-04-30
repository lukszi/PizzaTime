import cv2 as cv
import argparse
import numpy as np

from number_classifier.generate_data_set import read_label_csv
from opencv.extract_number_images import extract_ordered_numbers

max_value = 255
max_value_H = 360 // 2
low_H = 0
low_S = 0
low_V = 0
alpha = 33
beta = 0
high_H = max_value_H
high_S = max_value
high_V = max_value
window_capture_name = 'Video Capture'
window_detection_name = 'Object Detection'
low_H_name = 'Low H'
low_S_name = 'Low S'
low_V_name = 'Low V'
alpha_name = 'alpha'
beta_name = "beta"
high_H_name = 'High H'
high_S_name = 'High S'
high_V_name = 'High V'


def on_low_H_thresh_trackbar(val):
    global low_H
    global high_H
    low_H = val
    low_H = min(high_H - 1, low_H)
    cv.setTrackbarPos(low_H_name, window_detection_name, low_H)


def on_alpha_trackbar(val):
    global alpha
    alpha = val
    cv.setTrackbarPos(alpha_name, window_detection_name, alpha)


def on_beta_trackbar(val):
    global beta
    beta = val
    cv.setTrackbarPos(beta_name, window_detection_name, beta)


def on_high_H_thresh_trackbar(val):
    global low_H
    global high_H
    high_H = val
    high_H = max(high_H, low_H + 1)
    cv.setTrackbarPos(high_H_name, window_detection_name, high_H)


def on_low_S_thresh_trackbar(val):
    global low_S
    global high_S
    low_S = val
    low_S = min(high_S - 1, low_S)
    cv.setTrackbarPos(low_S_name, window_detection_name, low_S)


def on_high_S_thresh_trackbar(val):
    global low_S
    global high_S
    high_S = val
    high_S = max(high_S, low_S + 1)
    cv.setTrackbarPos(high_S_name, window_detection_name, high_S)


def on_low_V_thresh_trackbar(val):
    global low_V
    global high_V
    low_V = val
    low_V = min(high_V - 1, low_V)
    cv.setTrackbarPos(low_V_name, window_detection_name, low_V)


def on_high_V_thresh_trackbar(val):
    global low_V
    global high_V
    high_V = val
    high_V = max(high_V, low_V + 1)
    cv.setTrackbarPos(high_V_name, window_detection_name, high_V)


def search_for_best_parameters():
    accuracys = np.array([])
    for low_H in range(100, 160, 10):
        print(low_H)
        max_H = 180
        for low_S in range(32, 100, 10):
            for max_S in range(180, 255, 10):
                for low_V in range(80, 140, 10):
                    max_V = 255
                    file_data = read_label_csv("../res/data/base_data/labels.csv")
                    accuracy = 0
                    for file_name in file_data:
                        expected_len = len(file_data[file_name])
                        gotten_len = len(extract_ordered_numbers(f"../res/data/base_data/{file_name}"))
                        if expected_len == gotten_len:
                            accuracy += 1
                    accuracy = accuracy / 392
                    accuracys = np.append(accuracys, np.array([accuracy, low_H, max_H, low_S, max_S, low_V, max_V]))
    best_index = np.argmax(accuracys, axis=0)
    return accuracys[best_index]



# print(search_for_best_parameters())

parser = argparse.ArgumentParser(description='Code for Thresholding Operations using inRange tutorial.')
parser.add_argument('--camera', help='Camera divide number.', default=0, type=int)
args = parser.parse_args()
cap = cv.VideoCapture(args.camera)
cv.namedWindow(window_capture_name)
cv.namedWindow(window_detection_name)
cv.createTrackbar(low_H_name, window_detection_name, low_H, max_value_H, on_low_H_thresh_trackbar)
cv.createTrackbar(high_H_name, window_detection_name, high_H, max_value_H, on_high_H_thresh_trackbar)
cv.createTrackbar(low_S_name, window_detection_name, low_S, max_value, on_low_S_thresh_trackbar)
cv.createTrackbar(high_S_name, window_detection_name, high_S, max_value, on_high_S_thresh_trackbar)
cv.createTrackbar(low_V_name, window_detection_name, low_V, max_value, on_low_V_thresh_trackbar)
cv.createTrackbar(high_V_name, window_detection_name, high_V, max_value, on_high_V_thresh_trackbar)
cv.createTrackbar(alpha_name, window_detection_name, alpha, 100, on_alpha_trackbar)
cv.createTrackbar(beta_name, window_detection_name, beta, 100, on_beta_trackbar)
im = cv.imread("../res/data/base_data/1.jpg")
im = cv.resize(im, (640, 480))

counter = 1
while True:

    frame = im
    if frame is None:
        break

    if counter % 5000 == 0:
        new_image = np.zeros(im.shape, im.dtype)
        for y in range(im.shape[0]):
            for x in range(im.shape[1]):
                for c in range(im.shape[2]):
                    new_image[y, x, c] = np.clip(alpha/100*3 * im[y, x, c] + beta, 0, 255)
        frame = new_image
        counter = 0
    frame_HSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    frame_threshold = cv.inRange(frame_HSV, (low_H, low_S, low_V), (high_H, high_S, high_V))

    cv.imshow(window_capture_name, frame)
    cv.imshow(window_detection_name, frame_threshold)

    counter += 1
    key = cv.waitKey(30)
    if key == ord('q') or key == 27:
        break
