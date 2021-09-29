
import numpy as np


def evaluate_metrics(y_true, y_pred, num_of_classes=2): # num_of_classes: 1) 'background' and 2) 'marked ROI'
    
    class_wise_iou_coeff = []
    class_wise_dice_score = []
    
    y_true = np.around(y_true)
    y_pred = np.around(y_pred)

    tol_val = 0.00001 # Define tolerance val
    for i in range(num_of_classes):

        intersection = np.sum((y_pred == i) * (y_true == i))
        y_true_area = np.sum((y_true == i))
        y_pred_area = np.sum((y_pred == i))
        combined_area = y_true_area + y_pred_area

        iou = (intersection + tol_val) / (combined_area - intersection + tol_val)
        class_wise_iou_coeff.append(iou)

        dice_score =  2 * ((intersection + tol_val) / (combined_area + tol_val))
        class_wise_dice_score.append(dice_score)
    
    return class_wise_dice_score, class_wise_iou_coeff
