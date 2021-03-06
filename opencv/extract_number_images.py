from typing import Tuple

import cv2
import numpy as np


IMAGE_PATH = "../res/data/base_data/7.jpg"
NN_IMAGE_SIZE = (50, 50)


def get_image(image_path: str = IMAGE_PATH):
    im = cv2.imread(image_path)
    # return cv2.resize(im, (640, 480))
    return im


def threshold_image(image,
                    h: Tuple[int, int] = (100, 180),
                    s: Tuple[int, int] = (26, 255),
                    v: Tuple[int, int] = (130, 255)):
    (low_h, high_h) = h
    (low_s, high_s) = s
    (low_v, high_v) = v

    image = cv2.GaussianBlur(image, (5, 5), cv2.BORDER_DEFAULT)
    frame_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    frame_threshold = cv2.inRange(frame_hsv, (low_h, low_s, low_v), (high_h, high_s, high_v))
    return frame_threshold


def are_contours_acceptable(contours):
    contours_result = []
    for contour, index in zip(contours, range(len(contours))):
        size = cv2.contourArea(contour)
        if size > 500:
            for contour_before in contours_result:
                for values in contour:
                    for values_before in contour_before:
                        if values[0][0] > values_before[0][0]:
                            break
            contours_result.append(contour)
    return contours_result


def find_contours(image, thresholded_image):
    contours, hierarchy = cv2.findContours(thresholded_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # filter out smaller contours
    for contour in contours:
        print(cv2.contourArea(contour))
    print('------')
    # contours = list(filter(lambda contour: cv2.contourArea(contour) > 500, contours))
    contours = are_contours_acceptable(contours)
    print(len(contours))
    # draw_contours(contours, image)
    return contours


def get_surrounding_contour(contours, image_dim: Tuple[int, int]):
    max_x = 0
    max_y = 0
    min_x = None
    min_y = None

    # Figure out minimal Box that bounds all of the given contours
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        x_top = x + w
        y_top = y + h

        max_x = x_top if x_top > max_x else max_x
        max_y = y_top if y_top > max_y else max_y
        min_x = x if not min_x or x < min_x else min_x
        min_y = y if not min_y or y < min_y else min_y

    # Add a Buffer onto that
    # Todo


def draw_contours(contours, image):
    for contour in contours:
        if cv2.contourArea(contour) > 500:
            [x, y, w, h] = cv2.boundingRect(contour)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow('contours2', image)
    cv2.waitKey(0)


def extract_image(image, contour):
    y = contour[1]
    y_max = y + contour[3]

    x = contour[0]
    x_max = x + contour[2]

    extract = image[y:y_max, x:x_max]
    return extract


def crop_contours(image, contours):
    images = []
    for contour in contours:
        [x, y, w, h] = cv2.boundingRect(contour)
        extracted_image = extract_image(image, [x, y, w, h])
        images.append({"bounding_rect": (x, y, w, h), "image": extracted_image})
    return images


def order_cropped_contours(contours: list):
    contours.sort(key=lambda contour: contour["bounding_rect"][0])
    return contours


def extract_ordered_numbers(image_path: str):
    image = get_image(image_path)

    # Extract numbers from image
    preprocessed_image = threshold_image(image)
    # cv2.imshow('contours1', preprocessed_image)
    # cv2.waitKey(0)
    contours = find_contours(preprocessed_image, preprocessed_image)
    contour_crops = crop_contours(image, contours)
    order_cropped_contours(contour_crops)
    return contour_crops


def preprocess_image_for_nn_classification(image):
    image = cv2.resize(image, NN_IMAGE_SIZE)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)



if __name__ == '__main__':
    im = get_image()

    # Extract numbers from image
    preprocessed_im = threshold_image(im)
    cnt = find_contours(im, preprocessed_im)
    get_surrounding_contour(cnt, preprocessed_im.shape)
    cnt_im = crop_contours(im, cnt)
    draw_contours(cnt, im)
