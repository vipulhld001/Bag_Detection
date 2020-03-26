import cv2
import tensorflow as tf
import keras

CATOGORIES = ["nbag", "bag"]


def prepare(filepath):
    IMG_SIZE = 80
    img_array = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)


model = tf.keras.models.load_model("64by2and80by803-CNN.model")

prediction = model.predict([prepare('sss.jpg')])

print(CATOGORIES[int(prediction[0][0])])
