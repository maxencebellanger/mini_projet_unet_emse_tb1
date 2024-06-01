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

train_root  = '../training/'
test_root   = '../test/test_images/'
predictions_root = '../test/predicted_masks/'

images = os.listdir(train_root+"images")
tests = os.listdir(test_root+"images")
preidctions_path = os.listdir(predictions_root)
nb_train_samples = len(images)
nb_test_samples = len(tests)

images_train = np.zeros((nb_train_samples,HEIGHT,WIDTH,1))
masks_train = np.zeros((nb_train_samples,HEIGHT,WIDTH,1))
images_test = np.zeros((nb_test_samples,HEIGHT,WIDTH,1))
masks_test = np.zeros((nb_test_samples,HEIGHT,WIDTH,1))
predictions = np.zeros((nb_test_samples,HEIGHT,WIDTH,1))


for i in range(nb_train_samples):
  path = images[i]
  img = cv2.imread(train_root+"images/"+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  images_train[i] = img[:,:,1].reshape(512,512,1)
  img = cv2.imread(train_root+'masks/'+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  masks_train[i] = img[:,:,1].reshape(512,512,1)


for i in range(nb_test_samples):
  path = tests[i]
  img = cv2.imread(test_root+"images/"+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  images_test[i] = img[:,:,1].reshape(512,512,1)
  img = cv2.imread(test_root+'masks/'+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  masks_test[i] = img[:,:,1].reshape(512,512,1)
  path = preidctions_path[i]
  img = cv2.imread(predictions_root+path)
  img = cv2.resize(img, (HEIGHT,WIDTH))
  predictions[i] = img[:,:,1].reshape(512,512,1)
  
  
images_train /= 255.0
images_test /= 255.0
masks_train /= 255.0

np.save('../model_files/train_images_array.npy',images_train)
np.save('../model_files/train_masks_array.npy',masks_train)
np.save('../model_files/test_images_array.npy',images_test)
np.save('../model_files/test_masks_array.npy',masks_test)
np.save('../model_files/predictions_array.npy',predictions)
