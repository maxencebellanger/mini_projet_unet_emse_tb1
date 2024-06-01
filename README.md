# mini_projet_unet_emse_tb1

All thanks to sagnik1511, I used his project to start : https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras 

This project is an assingment in a course at EMSE: Introduction to image processing.

## 1 - Setup
### 1.1 - Training data
Make a directory named : *training* which contains directories named *images* and *masks*. It will contain all the training data.

### 1.2 - Test data
Make a directory named: *test* which contains directories named *predicted_masks* and *test_images*. <br>
*test_images* will contain two folders named *images* and *masks* like the *training* folder. 

### 1.3 - Images
Your images should have a size of 512x512 pixels and be RGB.

## 2 - Running the Unet
### 2.1 - Data initialization
Run *UNET_data.py* to create files that will contain all the training and test data.

### 2.2 - Training
Run *UNET_train.py* to train the model.

### 2.3 - Prediction
Run *UNET_predict.py* to predict the masks of your images. <br>
The results appears in *test/predicted_masks*. If your images are all black or all white you should modify the threshold in *UNET_predict.py*.

### 2.4 - Performance 
The file *rapport.py* contains functions to test the performance of the model. 