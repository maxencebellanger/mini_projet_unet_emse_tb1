import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def predict_masks():
    model = keras.models.load_model("../model_files/Unet.keras")
    test_images = np.load('../model_files/test_images_array.npy')
    predicted_masks = model.predict(test_images)
    for i in range(len(test_images)):
        predicted_masks[i] = predicted_masks[i] > 0.5
        cv2.imwrite("../test/predicted_masks/"+str(i)+".png", predicted_masks[i]*255)


predict_masks()