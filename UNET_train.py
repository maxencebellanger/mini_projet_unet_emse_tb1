# -*- coding: utf-8 -*-
"""
Created by @sagnik1511 (https://github.com/sagnik1511). 
All rights reserved.

"""

from UNET_lowered import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.layers import Input , Conv2D , MaxPooling2D , Dropout , concatenate , UpSampling2D
from tensorflow.keras import models
from tensorflow.keras import losses
from tensorflow.keras import optimizers
import numpy as np

model = UNet((512,512,1))

images_train = np.load('../model_files/train_images_array.npy')
masks_train = np.load('../model_files/train_masks_array.npy')

EPOCHS = 15
#VAL_DATA = (X_val,y_val)
BATCH_SIZE = 10
callbacks=[keras.callbacks.ModelCheckpoint('Unet_XRAY_best.h5.keras',save_best_only=True)]


model.compile(optimizer = optimizers.Adam(1e-4) , 
              loss = losses.BinaryCrossentropy(from_logits = False),
              metrics = ['accuracy'])

history = model.fit(images_train, masks_train, 
                    batch_size = BATCH_SIZE,
                    epochs = EPOCHS , callbacks = callbacks ,
                    verbose = 1)#, validation_data = VAL_DATA)

model.save("../model_files/Unet.keras")