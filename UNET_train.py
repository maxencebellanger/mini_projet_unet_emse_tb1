# -*- coding: utf-8 -*-
from UNET_lowered import *
from UNET_data import save_training_data
import tensorflow as tf
from tensorflow import keras
import numpy as np

model = UNet_lowered((512,512, 1))

save_training_data()

images_train = np.load('../model_files/train_images_array.npy')
masks_train = np.load('../model_files/train_masks_array.npy')

EPOCHS = 15
#VAL_DATA = (X_val,y_val)
BATCH_SIZE = 10
callbacks=[keras.callbacks.ModelCheckpoint('Unet_XRAY_best.h5.keras',save_best_only=True)]


model.compile(optimizer = keras.optimizers.Adam(1e-4) , 
              loss = keras.losses.BinaryCrossentropy(from_logits = False),
              metrics = ['accuracy'])

history = model.fit(images_train, masks_train, 
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS , callbacks = callbacks ,
                    verbose = 1)#, validation_data = VAL_DATA)

model.save("../model_files/Unet.keras")