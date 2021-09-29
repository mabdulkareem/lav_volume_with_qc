
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os


from general_utilities import create_dir
from compute_metrics_seg import evaluate_metrics


def display_images(images, pred_mask, true_mask, save_img_path, all_eval=False, per_epoch=False): 
    
    save_img_path = save_img_path + '/model_predictions_using_eval_data'
    
    if all_eval: 
        save_img_path = save_img_path + '_all'
        
    if not per_epoch: 
        create_dir(save_img_path)
        
    if per_epoch: 
        save_img_path = save_img_path.replace('/model_predictions_using_eval_data', '')
        parent_folder, _ = os.path.split(save_img_path)
        create_dir(parent_folder)

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for j in range(len(images)): 
        display_item = images[j, ...], true_mask[j, ...], pred_mask[j, ...]
        
        plt.figure(figsize=(15, 6))
        for i in range(len(title)):
            plt.subplot(1, len(title), i+1)
            plt.title(title[i])
            plt.xticks([])
            plt.yticks([])
            if i == 0: 
                image_to_display = display_item[i]
            else: 
                image_to_display = np.around(display_item[i])
            #plt.imshow(tf.keras.preprocessing.image.array_to_img(image_to_display))
            # Rotate Nifti to normal way of visualization: 
            plt.imshow(tf.keras.preprocessing.image.array_to_img(np.rot90(image_to_display,3))) 

        if per_epoch == False: 
            class_wise_dice_score, class_wise_iou_coeff = evaluate_metrics(display_item[1], display_item[2])
            string_1 = 'Background - Dice Score: {0:.4f}, IOU: {0:.4f} '.format(class_wise_dice_score[0], class_wise_iou_coeff[0])
            string_2 = 'Marked ROI - Dice Score: {0:.4f}, IOU: {0:.4f} '.format(class_wise_dice_score[1], class_wise_iou_coeff[1])
            plt.xlabel('\n' + string_1 + '\n\n' + string_2)
        
        if per_epoch: 
            plt.savefig(save_img_path + '.png')
        else: 
            plt.savefig(save_img_path + '/model_predictions_using_eval_data_' + str('{0:04d}'.format(j)) + '.png')
        plt.close()

        
def create_mask(pred_mask):
    '''
    Get the channel with the highest probability to create the segmentation mask. Recall that the output of 
    our UNet has 2 channels. Thus, for each pixel, the predicition will be the channel with the highest probability.
    '''
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    return pred_mask
        

def pred_plots_for_evaluation(eval_dataset, model, NO_OF_IMAGES, save_img_path, all_eval=False):    
    
    dataset = eval_dataset
    
    for image, true_mask, _ in dataset:
        
        pred_semi_mask = model(image) # 'semi' because it still has '2 channels' as specified as output of the UNet model
        pred_semi_mask_batches = pred_semi_mask.numpy() 
        true_mask_batches = true_mask.numpy() 
        image_batches = image.numpy()
    
    # Create an empty array with the right dimension
    pred_mask_batches = np.zeros_like(pred_semi_mask_batches)
    pred_mask_batches = pred_mask_batches[..., 0]
    pred_mask_batches = pred_mask_batches[..., np.newaxis]
    
    for i in range(len(pred_mask_batches)): 
        a = pred_semi_mask_batches[i,...]
        b = create_mask(a)
        pred_mask_batches[i, ...] = b # we now have '1 channel'
        
    # generate 'size' random numbers in range 0 to 'length of y_pred_batches': 
    indexes = np.random.choice(len(pred_mask_batches), size=NO_OF_IMAGES)
    
    image_to_plot = image_batches[indexes]
    true_mask_to_plot = true_mask_batches[indexes]
    pred_mask_to_plot = pred_mask_batches[indexes]
    
    display_images(image_to_plot, pred_mask_to_plot, true_mask_to_plot, save_img_path, all_eval=all_eval)
