import csv
import os

import cv2

from opencv.extract_number_images import extract_ordered_numbers

CSV_FILENAME_POS = 0
CSV_LEFT_NUMBER_POS = 1
CSV_RIGHT_NUMBER_POS = 2
LABEL_FILE_NAME = "labels.csv"


def write_labels(label_extraction_map, base_target_folder: str):
    for file in label_extraction_map:
        target_folder = f"{base_target_folder}/{file}"
        # create folder
        os.mkdir(target_folder)
        extracted_images = label_extraction_map[file]["ordered_numbers"]
        labels = label_extraction_map[file]["labels"]
        for i in range(0, min(len(labels), len(extracted_images))):
            image = extracted_images[i]["image"]
            label = labels[i]
            file_path = f"{target_folder}/{i}-{label}.jpg"
            cv2.imwrite(file_path, image)


def generate_data_set(source_folder: str = "../res/data/base_data", target_folder: str = "../res/data/generated"):
    labels = read_label_csv(f"{source_folder}/{LABEL_FILE_NAME}")
    label_extraction_map = {}
    for file in labels:
        ordered_numbers = extract_ordered_numbers(f"{source_folder}/{file}")
        expected_len = len(labels[file])
        gotten_len = len(ordered_numbers)
        if expected_len == gotten_len:
            label_extraction_map[file] = {"ordered_numbers": ordered_numbers, "labels": labels[file]}
    write_labels(label_extraction_map, target_folder)


def read_label_csv(label_file_path: str = "res/data/base_data/labels.csv"):
    labels = {}
    with open(label_file_path, "r", encoding="UTF-8") as label_file_path:
        label_reader = csv.reader(label_file_path)
        for csv_label in label_reader:
            labels[csv_label[CSV_FILENAME_POS]] = list(csv_label[CSV_LEFT_NUMBER_POS] + csv_label[CSV_RIGHT_NUMBER_POS])

    return labels


def filter_wrong():
    file_data = read_label_csv("../res/data/base_data/labels.csv")
    training_files = []
    for file_name in file_data:
        expected_len = len(file_data[file_name])
        gotten_len = len(extract_ordered_numbers(f"../res/data/base_data/{file_name}"))
        if expected_len == gotten_len:
            training_files.append(expected_len)
    return training_files


if __name__ == '__main__':
    generate_data_set()
    # print(len(filter_wrong()))