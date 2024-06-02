# -*- coding: utf-8 -*-
from UNET_data import save_test_data, save_predictions_data
from tensorflow import keras
import numpy as np
import cv2
import skimage as sk

def predict_masks():
    model = keras.models.load_model("../model_files/Unet.keras")
    test_images = np.load('../model_files/test_images_array.npy')
    predicted_masks = model.predict(test_images)
    for i in range(len(test_images)):
        predicted_masks[i] = predicted_masks[i] > 0.45
        cv2.imwrite("../test/predicted_masks/"+str(i)+".png", predicted_masks[i]*255)
        cv2.imwrite("../test/enhanced_predicted_masks/"+str(i)+".png", enhance_prediction(predicted_masks[i])*255)



def enhance_prediction(prediction):
    img = prediction.reshape(512, 512) > 0
    img = sk.morphology.binary_opening(img, sk.morphology.disk(5))
    img = sk.morphology.remove_small_objects(img, 10)
    img = sk.morphology.binary_closing(img, sk.morphology.disk(5))
    return img


save_test_data()
predict_masks()
save_predictions_data()
