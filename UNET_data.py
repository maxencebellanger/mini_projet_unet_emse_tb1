# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os

HEIGHT = 512
WIDTH = 512

train_root  = '../training/'
test_root   = '../test/test_images/'
predictions_root = '../test/predicted_masks/'
enhanced_predictions_root = '../test/enhanced_predicted_masks/'


def save_training_data():
  """
    Store the training data into npy files and make it readable for the model
  """
  images = os.listdir(train_root+"images")
  nb_train_samples = len(images)

  images_train = np.zeros((nb_train_samples,HEIGHT,WIDTH,1))
  masks_train = np.zeros((nb_train_samples,HEIGHT,WIDTH,1))

  for i in range(nb_train_samples):
    path = images[i]
    img = cv2.imread(train_root+"images/"+path)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    images_train[i] = img[:,:,1].reshape(512,512,1)
    img = cv2.imread(train_root+'masks/'+path)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    masks_train[i] = img[:,:,1].reshape(512,512,1)

  images_train /= 255.0
  masks_train /= 255.0

  np.save('../model_files/train_images_array.npy',images_train)
  np.save('../model_files/train_masks_array.npy',masks_train)

def save_test_data():
  """
    Store the test data into npy files to make it easier to test
  """
  tests = os.listdir(test_root+"images")
  nb_test_samples = len(tests)
  
  images_test = np.zeros((nb_test_samples,HEIGHT,WIDTH,1))
  masks_test = np.zeros((nb_test_samples,HEIGHT,WIDTH,1))

  for i in range(nb_test_samples):
    path = tests[i]
    img = cv2.imread(test_root+"images/"+path)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    images_test[i] = img[:,:,1].reshape(512,512,1)
    img = cv2.imread(test_root+'masks/'+path)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    masks_test[i] = img[:,:,1].reshape(512,512,1)

  images_test /= 255.0

  np.save('../model_files/test_images_array.npy',images_test)
  np.save('../model_files/test_masks_array.npy',masks_test)

def save_predictions_data():
  """
    Store the prediction data into npy files to make it easier to evaluate
  """
  predictions_images = os.listdir(predictions_root)
  nb_predictions = len(predictions_images)
  preidctions_path = os.listdir(predictions_root)
  predictions = np.zeros((nb_predictions,HEIGHT,WIDTH,1))
  enhanced_predictions = np.zeros((nb_predictions,HEIGHT,WIDTH,1))

  for i in range(nb_predictions):
    path = preidctions_path[i]
    img = cv2.imread(predictions_root+path)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    predictions[i] = img[:,:,1].reshape(512,512,1)

  for i in range(nb_predictions):
    path = preidctions_path[i]
    img = cv2.imread(enhanced_predictions_root+path)
    img = cv2.resize(img, (HEIGHT,WIDTH))
    enhanced_predictions[i] = img[:,:,1].reshape(512,512,1)

  np.save('../model_files/predictions_array.npy',predictions)
  np.save('../model_files/enhanced_predictions_array.npy',enhanced_predictions)

def save_all_data():
  save_training_data()
  save_test_data()
  save_predictions_data()