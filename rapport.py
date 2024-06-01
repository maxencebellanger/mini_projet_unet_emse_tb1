import numpy as np
from sklearn.metrics import confusion_matrix

#Un script qui présente les résultats du modèles :
#Affiche quelques images avec les prédictions et les vraies valeurs
#Affiche la mtriice de confusion
#Fait une analyse de la performance du modèle : distrubution de l'aire, des périmètres, de chaque forme prédite sur les masks


import matplotlib.pyplot as plt

# Function to display images with predictions and true values
def display_images(images, predictions, true_values):
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(len(images)):
        axes[i].imshow(images[i])
        axes[i].set_title(f"Prediction: {predictions[i]}, True Value: {true_values[i]}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

# Function to calculate and display confusion matrix
def display_confusion_matrix(true_values, predictions):
    cm = confusion_matrix(true_values, predictions)
    classes = np.unique(true_values)
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title='Confusion Matrix',
           ylabel='True label',
           xlabel='Predicted label')

    plt.show()

# Function to analyze model performance
def analyze_performance(predictions, masks):
    # Calculate area and perimeter for each predicted shape
    areas = []
    perimeters = []
    for mask in masks:
        area = np.sum(mask)
        perimeter = np.sum(mask) - 4 * np.sum(mask[1:-1, 1:-1]) - 2 * np.sum(mask[0, 1:-1]) - 2 * np.sum(mask[-1, 1:-1]) - 2 * np.sum(mask[1:-1, 0]) - 2 * np.sum(mask[1:-1, -1])
        areas.append(area)
        perimeters.append(perimeter)

    # Plot distribution of areas and perimeters
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(areas, bins=20)
    axes[0].set_title('Distribution of Areas')
    axes[0].set_xlabel('Area')
    axes[0].set_ylabel('Count')

    axes[1].hist(perimeters, bins=20)
    axes[1].set_title('Distribution of Perimeters')
    axes[1].set_xlabel('Perimeter')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.show()


# Load the images, predictions, and true values
images = np.load('../model_files/test_images_array.npy') 
predictions = np.load('../model_files/predictions_array.npy') 
masks = np.load('../model_files/test_masks_array.npy') 

# Display images with predictions and true values
#display_images(images, predictions, masks)

# Display confusion matrix
#display_confusion_matrix(masks, predictions)

# Analyze model performance
analyze_performance(predictions, masks)

