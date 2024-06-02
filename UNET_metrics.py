import numpy as np
from UNET_data import *
import skimage as sk
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

report_path = "../metrics/"

def display_images(images, predictions, enhanced_predictions, masks, images_to_display=5):
    for i in range(min(len(images), images_to_display)):
        plt.figure(i, figsize=(20, 6))
        
        plt.subplot(1,4,1)
        plt.imshow(images[i], cmap='gray')
        plt.title("Original image")
        plt.axis('off')

        plt.subplot(1,4,2)
        plt.imshow(masks[i], cmap='gray')
        plt.title("True mask")
        plt.axis('off')

        plt.subplot(1,4,3)
        plt.imshow(predictions[i], cmap='gray')
        plt.title("Predicted mask")
        plt.axis('off')

        plt.subplot(1,4,4)
        plt.imshow(enhanced_predictions[i], cmap='gray')
        plt.title("Enhanced predicted mask")
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(report_path+"images_comparaison"+str(i))

    #plt.show()

def calculate_metrics(TN, FP, FN, TP):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    iou = TP / (TP + FP + FN)
    dice = 2 * TP / (2 * TP + FP + FN)
    
    return accuracy, precision, recall, f1_score, iou, dice

# Function to calculate and display confusion matrix
def display_confusion_matrix(true_values, predictions, enhanced_predictions):
    true_values = true_values.flatten() > 0
    predictions = predictions.flatten() > 0
    enhanced_predictions = enhanced_predictions.flatten() > 0

    cm_p = confusion_matrix(true_values, predictions)
    cm_ep = confusion_matrix(true_values, enhanced_predictions)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm_p, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title('Confusion matrix (Prediction)')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Real')

    sns.heatmap(cm_ep, annot=True, fmt="d", cmap="Blues", ax=axes[1])
    axes[1].set_title('Confusion matrix (Enhanced Predictions)')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Real')

    plt.tight_layout()
    plt.savefig(report_path+"confusion_matrices")

    ac_p, pr_p, re_p, f1_p, iou_p, dice_p = calculate_metrics(cm_p[0,0], cm_p[0,1], cm_p[1,0], cm_p[1,1])
    ac_ep, pr_ep, re_ep, f1_ep, iou_ep, dice_ep = calculate_metrics(cm_ep[0,0], cm_ep[0,1], cm_ep[1,0], cm_ep[1,1])
    #plt.show()

    return ac_p, pr_p, re_p, f1_p, iou_p, dice_p, ac_ep, pr_ep, re_ep, f1_ep, iou_ep, dice_ep


def geometrics_metrics(predictions, enhanced_predictions, masks):
    perimeters_predictions = np.array([])
    areas_predictions = np.array([])
    perimeters_masks = np.array([])
    areas_masks = np.array([])
    perimeters_enhanced_predictions = np.array([])
    areas_enhanced_predictions = np.array([])

    for i in range(len(predictions)):
        prediction_i = predictions[i].reshape(512, 512) > 0
        mask_i = masks[i].reshape(512,512) > 0
        enhanced_prediction_i = enhanced_predictions[i].reshape(512, 512) > 0

        prediction_i, nb_object_prediction = sk.measure.label(prediction_i, return_num=True)
        mask_i, nb_object_mask = sk.measure.label(mask_i, return_num=True)
        enhanced_prediction_i, nb_object_enhanced_prediction = sk.measure.label(enhanced_prediction_i, return_num=True)

        perimeters_prediction_i = np.zeros(nb_object_prediction)
        perimeters_mask_i = np.zeros(nb_object_mask)
        perimeters_enhanced_prediction_i = np.zeros(nb_object_enhanced_prediction)

        areas_prediction_i = np.zeros(nb_object_prediction)
        areas_mask_i = np.zeros(nb_object_mask)
        areas_enhanced_prediction_i = np.zeros(nb_object_enhanced_prediction)

        for k in range(nb_object_prediction):
            perimeters_prediction_i[k] = sk.measure.perimeter(prediction_i == k)
            areas_prediction_i[k] = np.sum(prediction_i == k)

        for k in range(nb_object_mask):
            perimeters_mask_i[k] = sk.measure.perimeter(mask_i == k)
            areas_mask_i[k] = np.sum(mask_i == k)

        for k in range(nb_object_enhanced_prediction):
            perimeters_enhanced_prediction_i[k] = sk.measure.perimeter(enhanced_prediction_i == k)
            areas_enhanced_prediction_i[k] = np.sum(enhanced_prediction_i == k)

        perimeters_predictions = np.append(perimeters_predictions, perimeters_prediction_i)
        areas_predictions = np.append(areas_predictions, areas_prediction_i)
        perimeters_masks = np.append(perimeters_masks, perimeters_mask_i)
        areas_masks = np.append(areas_masks, areas_mask_i)
        perimeters_enhanced_predictions = np.append(perimeters_enhanced_predictions, perimeters_enhanced_prediction_i)
        areas_enhanced_predictions = np.append(areas_enhanced_predictions, areas_enhanced_prediction_i)

    return perimeters_predictions, areas_predictions, perimeters_enhanced_predictions, areas_enhanced_predictions, perimeters_masks, areas_masks


def display_geometrics_metrics(predictions, enhanced_predictions, masks):
    # Calculate area and perimeter for each predicted shape

    perimeters_prediction, areas_prediction, perimeters_enhanced_predictions, areas_enhanced_predictions, perimeters_mask, areas_mask = geometrics_metrics(predictions, enhanced_predictions, masks)

    #print(perimeters_prediction, perimeters_mask)

    # Plot distribution of areas and perimeters
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.kdeplot(perimeters_prediction, label='Predicted objects', ax=ax[0])
    sns.kdeplot(perimeters_enhanced_predictions, label='Enhanced predicted objects', ax=ax[0])
    sns.kdeplot(perimeters_mask, label='True objects', ax=ax[0])
    ax[0].set_title('Distribution of perimeters')
    ax[0].set_xscale('log')
    ax[0].set_xlabel('Perimeter') 
    ax[0].set_ylabel('Densité')
    ax[0].legend()

    sns.kdeplot(areas_prediction, label='Predicted objects', ax=ax[1])
    sns.kdeplot(areas_enhanced_predictions, label='Enhanced predicted objects', ax=ax[1])
    sns.kdeplot(areas_mask, label='True objects', ax=ax[1])
    ax[1].set_title('Distribution of areas')
    ax[1].set_xlabel('Area')
    ax[1].set_ylabel('Densité')
    ax[1].set_xscale('log')
    ax[1].legend()
    
    fig.tight_layout()
    fig.savefig(report_path+"geometrics_metrics")
    #plt.show()


# Load data
images = np.load('../model_files/test_images_array.npy') 
predictions = np.load('../model_files/predictions_array.npy') 
masks = np.load('../model_files/test_masks_array.npy') 
enhanced_predictions = np.load('../model_files/enhanced_predictions_array.npy')

# Display images with predictions and true values
display_images(images, predictions, enhanced_predictions, masks)

# Display confusion matrix and metrics

ac_p, pr_p, re_p, f1_p, iou_p, dice_p, ac_ep, pr_ep, re_ep, f1_ep, iou_ep, dice_ep =display_confusion_matrix(masks, predictions, enhanced_predictions)
print("Metrics for predictions: ")
print("Accuracy: ", ac_p)
print("Precision: ", pr_p)
print("Recall: ", re_p)
print("F1 score: ", f1_p)
print("IoU: ", iou_p)
print("Dice: ", dice_p)

print("Metrics for enhanced predictions: ")
print("Accuracy:", ac_ep)
print("Precision: ", pr_ep)
print("Recall: ", re_ep)
print("F1 score: ", f1_ep)
print("IoU: ", iou_ep)
print("Dice: ", dice_ep)

# Analyze model performance
display_geometrics_metrics(predictions, enhanced_predictions, masks)

