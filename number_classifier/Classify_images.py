import cv2
import tensorflow as tf
import numpy as np

from number_classifier.Keras import reformat_image_for_nn

model = tf.keras.models.load_model('test.model') # Load Model


def classify_image(img):
    img = reformat_image_for_nn(img)
    img = np.array([np.expand_dims(img, -1)])
    prediction = model.predict([img])
    return np.argmax(prediction)


if __name__ == '__main__':
    img = cv2.imread('../res/data/generated/56.jpg/0-4.jpg')
    print(classify_image(img))