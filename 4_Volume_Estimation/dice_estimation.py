
import cv2
import numpy as np
from preprocess_image import image_resize

def compute_dice(y_true, y_pred, num_of_classes=2): # num_of_classes: 1) 'background' and 2) 'marked ROI'
    
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

        dice_score =  2 * ((intersection + tol_val) / (combined_area + tol_val))
        class_wise_dice_score.append(dice_score)
    
    return class_wise_dice_score[1]

def estimate_dice(seg_mask, list_transforms, list_warped_segmentations, IMG_DIM): 
    
    height, width = seg_mask.shape

    list_dice_scores = []
    for i in range(len(list_transforms)): 

        # Warp the segmentation with each transformation matrix
        transform_matrix = list_transforms[i]
        transformed_img = cv2.warpPerspective(seg_mask.numpy().astype('float32'), 
                                              transform_matrix, (width, height))

        # Compute Dice
        y_true = list_warped_segmentations[i]
        y_true = image_resize(y_true, IMG_DIM) # Masked image from NifTi that was then warped is resized here
        dice_score = compute_dice(y_true, transformed_img)
        list_dice_scores.append(dice_score)
        
    max_dice_value = max(list_dice_scores)
    
    return max_dice_value
