import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2

def predict_masks():
    model = keras.models.load_model("../model_files/Unet.keras")
    test_images = np.load('../model_files/test_image_array.npy')
    predicted_masks = model.predict(test_images)
    for i in range(len(test_images)):
        cv2.imwrite("../training/predicted_masks/"+str(i)+".png", predicted_masks[i])


predict_masks()