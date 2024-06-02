# -*- coding: utf-8 -*-
from UNET_model import *
from UNET_data import save_training_data
from tensorflow import keras
import numpy as np


save_training_data()

images_train = np.load('../model_files/train_images_array.npy')
masks_train = np.load('../model_files/train_masks_array.npy')

def train_unet(model, name):
    EPOCHS = 5
    BATCH_SIZE = 5
    callbacks=[keras.callbacks.ModelCheckpoint('Unet_XRAY_best.h5.keras',save_best_only=True)]


    model.compile(optimizer = keras.optimizers.Adam(1e-4) , 
                  loss = keras.losses.BinaryCrossentropy(from_logits = False),
                  metrics = ['accuracy'])

    history = model.fit(images_train, masks_train, 
                        batch_size = BATCH_SIZE,
                        epochs = EPOCHS , callbacks = callbacks ,
                        verbose = 1)

    model.save("../model_files/"+name+".keras")

#train_unet(UNet_lowered((512,512,1)), "Unet_low")
train_unet(UNet((512,512,1)), "Unet")