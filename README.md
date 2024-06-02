# mini_projet_unet_emse_tb1

All thanks to sagnik1511, I used his project to start : https://github.com/sagnik1511/U-Net-Reduced-with-TF-keras 

This project is an assingment in a course at EMSE: Introduction to image processing.

## 1 - Setup
### 1.1 - Python and packages

**Use python 3.8 for tensorflow compatibility** <br>

Install the packages needed with pip: <br>

```
pip install -r requirements.txt
```

You should use a virtual environment to avoid conflict with your actual installation.

```
python -m venv <virtual environment location>
```

### 1.2 - Folder organization 
Make a directory named : *training* which contains directories named *images* and *masks*. It will contain all the training data.

Make a directory named: *test* which contains directories named *predicted_masks* and *test_images*. <br>
*test_images* will contain two folders named *images* and *masks* like the *training* folder. 

You have to add a directory named *model_files* to store the model-related files: <br>

At the end your folder should look like this: 
```
├───mini_projet_unet_emse_tb1
├───model_files
├───test
│   ├───predicted_masks
│   └───test_images
│       ├───images
│       └───masks
└───training
    ├───images
    └───masks
```

### 1.4 - Data
Your images should have a size of 512x512 pixels and be RGB. <br>
Your masks should be binarized images.

## 2 - Running the Unet
### 2.1 - Training
Run *UNET_train.py* to train the model.

### 2.2 - Prediction
Run *UNET_predict.py* to predict the masks of your images. <br>
The results appears in *test/predicted_masks*. If your images are all black or all white you should modify the threshold in *UNET_predict.py*.

### 2.3 - Evaluate performance
The file *rapport.py* contains functions to measure performance of the model. 