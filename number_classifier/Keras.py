from os import walk, path

import numpy as np
import cv2
import tensorflow as tf

# Model / data parameters
from number_classifier.generate_data_set import read_label_csv

num_classes = 10
input_image_shape = (60, 30, 3)
data_point_size = input_image_shape[0] * input_image_shape[1]


def reformat_image_for_nn(img):
    img = np.resize(img, input_image_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype("float32") / 255
    img = np.array([np.expand_dims(img, -1)])
    return img

def get_traning_data(path_to_data: str = "../res/data/generated/"):
    file_data = read_label_csv("../res/data/base_data/labels.csv")
    y = np.array([[]])
    x = np.array([[]])
    index_of_data_point = 0
    for file_name in file_data:
        if path.exists(f'{path_to_data}{file_name}'):
            expected_numbers = file_data[file_name]
            iterator_for_expected_numbers = 0
            result_y = np.zeros(13)
            for (_, _, filenames) in walk(f'{path_to_data}{file_name}'):
                for file in filenames:
                    number_file = cv2.imread(f'{path_to_data}{file_name}/{file}')
                    number_file = reformat_image_for_nn(number_file)
                    x = np.append(x, number_file.reshape(-1))
                    number = expected_numbers[iterator_for_expected_numbers]
                    if number == 'E':
                        result_y[11] = 1
                    elif number == 'K':
                        result_y[12] = 1
                    elif number == 'Z':
                        result_y[10] = 1
                    else:
                        result_y[int(expected_numbers[iterator_for_expected_numbers])] = 1
                    y = np.append(y, result_y)
                    index_of_data_point += 1
    return x.reshape(int(len(x)/data_point_size), 60, 30), y.reshape(int(len(y)/13), 13)


if __name__ == '__main__':

    # the data, split between train and test sets
    x = cv2.imread('data/1.jpg')

    x_train, y_train = get_traning_data()

    # Scale images to the [0, 1] range
    # x_train = x_train.astype("float32") / 255
    # x_test = x_test.astype("float32") / 255
    # Make sure images have shape (28, 28, 1)
    # x_train = np.expand_dims(x_train, -1)
    # x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    # print(x_test.shape[0], "test samples")

    # convert class vectors to binary class matrices
    # y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    inputs = tf.keras.layers.Input(shape=(60, 30, 1))
    c = tf.keras.layers.Conv2D(128, (6, 6), padding="valid", activation=tf.nn.relu)(inputs)
    e = tf.keras.layers.Conv2D(64, (3, 3), padding="valid", activation=tf.nn.relu)(c)
    m = tf.keras.layers.MaxPool2D((2, 2), (2, 2))(e)
    f = tf.keras.layers.Flatten()(m)
    g = tf.keras.layers.Dense(60, activation=tf.nn.relu)(f)
    d = tf.keras.layers.Dense(20, activation=tf.nn.relu)(g)
    outputs = tf.keras.layers.Dense(13, activation=tf.nn.softmax)(d)

    model = tf.keras.models.Model(inputs, outputs)
    model.summary()


    # model = tf.keras.Sequential()
    # model.add(tf.keras.layers.Dense())
    # model.add(tf.keras.layers.Dense(100, activation=tf.nn.relu))
    # model.add(tf.keras.layers.Dense(12, activation=tf.nn.sigmoid))

    # print(model.input_shape)
    epochs = 50

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=epochs, validation_split=0.1, batch_size=1)


    model.save('test.model')

    # new_model = tf.keras.models.load_model('test.model') # Load Model

    # score = model.evaluate(x_test, y_test, verbose=0)
    # print("Test loss:", score[0])
    # print("Test accuracy:", score[1])
