# -*- coding: utf-8 -*-
"""
Created by @sagnik1511 (https://github.com/sagnik1511). 
All rights reserved.

After setting the images into specified folder directory,
we are going to take the images into the dataset
that will be used for training.

"""
import numpy as np
import cv2
import os

HEIGHT = 512
WIDTH = 512

train_root  = '../training/train_images/'
test_root   = '../training/test_images/'

images = os.listdir(train_root+"images")
tests = os.listdir(test_root)
nb_train_samples = len(images)
nb_test_samples = len(tests)

X_train = np.zeros((nb_train_samples,HEIGHT,WIDTH,1))
y_train = np.zeros((nb_train_samples,HEIGHT,WIDTH,1))
X_test = np.zeros((nb_test_samples,HEIGHT,WIDTH,1))


for i in range(nb_train_samples):
  path = images[i]
  img = cv2.imread(train_root+"images/"+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  X_train[i] = img[:,:,1].reshape(512,512,1)
  img = cv2.imread(train_root+'masks/'+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  y_train[i] = img[:,:,1].reshape(512,512,1)


for i in range(nb_test_samples):
  path = tests[i]
  img = cv2.imread(test_root+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  X_test[i] = img[:,:,1].reshape(512,512,1)
  
  
X_train /= 255.0
X_test /= 255.0
y_train /= 255.0



np.save('../model_files/train_image_array.npy',X_train)
np.save('../model_files/train_mask_array.npy',y_train)
np.save('../model_files/test_image_array.npy',X_test)