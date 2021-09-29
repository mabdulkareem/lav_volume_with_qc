

# Import General Packages: 
import os
import time
from pathlib import Path
from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

import warnings 
warnings.filterwarnings("ignore")


# Import User-defined Modules: 
from device_settings import gpu_memory_limit
from general_utilities import create_dir
from preprocess_image import preprocess, image_resize
from postprocess_image import create_mask
from dice_estimation import compute_dice, estimate_dice


def compute_segmentation_quality():
    # Device Settings: 
    # Limit memory of GPU to use 
    memory_limit = 18432         
    gpu_memory_limit(memory_limit)
    
    # Parameters: 
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
    
    # Data List Preparation: 
    # ----------------------    
    print('Starting to prepare patient data list...')
    
    # Prepare results destination directory
    create_dir(log_save_dir)
    
    # Define naming formats, create list of data files in path
    ct_data_nii_gz = '_CT.nii.gz'
    data_ct_nii_gz_list = glob(os.path.join(DATA_PATH, '*' + ct_data_nii_gz))
    data_ct_nii_gz_list.sort()
    
    print('Patient data list preparation completed. \n')
    print('\n', flush=True)
    # ---- End of Datasets List Preparation ----
    
    # Machine Learning Models' Loading 
    # --------------------------------
    print('Starting to load deep learning models ...')
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    loss = tf.keras.losses.MeanSquaredError(name='loss')
    
    # Load Identification Model
    model_ident = tf.keras.models.load_model(MODEL_IDENT_PATH, compile=False)
    model_ident.compile(optimizer=optimizer, loss=loss)
    
    # Load Segmentation Model
    model_seg = tf.keras.models.load_model(MODEL_SEG_PATH, compile=False)
    model_seg.compile(optimizer=optimizer, loss=loss)
    
    print('Loading of deep learning models completed. \n')
    print('\n', flush=True)
    # ---- End of Datasets List Preparation ----
    
    # Estimate Volume for Each Patient
    # ------------
    print('Segmentation quality estimation has started.')
    
    current_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print('Segmentation Quality Estimation Start Time: ', current_time)
    
    # Start Counting time for training
    start = time.perf_counter()
    
    # Directory to save images: 
    save_img_path = os.path.join(log_save_dir, 'images')
    create_dir(save_img_path)
    
    label_seg_list = []
    pred_seg_list = []
    title = ['Input Image', 'True Mask', 'Predicted Mask']
    j = 0
    no_seg = 2000
    for ct in data_ct_nii_gz_list: 
        
        loaded_img = nib.load(ct)
        img_array_ct = loaded_img.get_fdata()
        
        # To Be Used For Ground Truth Computation
        loaded_seg = nib.load(ct.replace('CT', 'LAV-label'))
        seg_array = loaded_seg.get_fdata()

        patient = os.path.split(ct)[-1] # Get patient file name
        patient = patient.replace(ct_data_nii_gz, '') # Get patient name: Remove the prefix "_CT.nii.gz"
        
        print('- - - - - - - - - - - - - - - - - - - - -')
        print('Patient: ', patient)
            
        print('\n', flush=True)
        
        img_shape_ct = img_array_ct.shape[2]
        for i in range(img_shape_ct): 
                        
            ct_image_i_ = img_array_ct[..., i]
            ct_image_i = preprocess(ct_image_i_, IMG_DIM)
            y_pred = model_ident(ct_image_i)
            y_pred = y_pred.numpy()
            
            # Convert numpy array element to a standard Python scalar, and then round: 
            y_pred = int(np.around(y_pred.item()))
            
            if y_pred == 1: # i.e. if slide needs to be contoured. 
                print('No. :', j)
                y_pred_seg = model_seg(ct_image_i)    # Output Size: [1, 224, 224, 2]
                y_pred_seg = create_mask(y_pred_seg)  # Output Size: [1, 224, 224, 1]
              
                # Estimate Dice
                pred_dice = estimate_dice(y_pred_seg[0, ..., 0], 
                                          list_transforms, 
                                          list_warped_segmentations, 
                                          IMG_DIM)
                print("Pred Dice: ", pred_dice)
                
                # Actual Dice
                seg_image_i = seg_array[..., i]
                ground_truth = image_resize(seg_image_i, IMG_DIM)
                actual_dice = compute_dice(ground_truth, y_pred_seg[0, ..., 0])
                print("Actual Dice: ", actual_dice)
                
                # Plot and save
                if j%20 == 0: # Save every 20th loop pass
                    
                    display_item = ct_image_i_, ground_truth, y_pred_seg[0, ..., 0]
                    plt.figure(figsize=(12, 6))
                    for i in range(len(title)):
                        plt.subplot(1, len(title), i+1)
                        plt.title(title[i])
                        plt.xticks([])
                        plt.yticks([])
                        image_to_display = display_item[i]
                        plt.imshow(np.rot90(image_to_display, 3))
                    plt.savefig(save_img_path + '/image_' + str('{0:05d}'.format(j)) + '.png')
                    plt.close()
                
                #Store results 
                label_seg_list.append(actual_dice)
                pred_seg_list.append(pred_dice)
                    
                if j == no_seg: 
                    break 
                    
                j += 1
        
                print('\n', flush=True)
            
        if j == no_seg: 
            break
     
    print('- - - - - - - - - - - - - - - - - - - - -')
    
    # Store times
    elapsed = time.perf_counter() - start
    print('Segmentation Quality Estimation Elapsed Time: %.3f seconds.' % elapsed)
    
    end_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print('Segmentation Quality Estimation Finish Time: ', end_time)
        
    # Store dataframe results in dataframe: 
    df_table = pd.DataFrame({'Label_Seg_Quality': label_seg_list, 
                             'Pred_Seg_Quality': pred_seg_list
                            })
        
    data_file_name = log_save_dir + 'results'
    df_table.to_csv(data_file_name + '.csv', encoding='utf-8', index=True)
    df_table.to_excel(data_file_name + '.xlsx', index=True) 
    
    print('\n', flush=True)
        
if __name__ == "__main__":
    
    
    # LAV VOLUME
    # -------------
    parent_directory = 'path/to/dataset/parent-directory/folder'
    
    # CT Image Data Path: 
    DATA_DIR_NAME = 'folder-name'
    DATA_PATH = os.path.join(parent_directory, DATA_DIR_NAME)
    
    # Path to Classfication Model: 
    ident_model = 'path/to/model-classification-folder/model_in_h5_format.h5'
    MODEL_IDENT_PATH = os.path.join(parent_directory, ident_model)
    
    # Path to Segmentation Model: 
    seg_model = 'path/to/model-segmentation-folder/model_in_h5_format.h5'
    MODEL_SEG_PATH = os.path.join(parent_directory, seg_model)
    
    # Define Path and Load Quality Control Transformation and Warped Seg. Matrices: 
    DATA_DESTINATION_FOLDER_NAME = 'Data_Quality_Control_LAV'
    npy_ct_file_name1 = os.path.join(os.path.join(parent_directory, DATA_DESTINATION_FOLDER_NAME), 
                                    'QC_LAV_Transform_Matrix.npy')
    npy_ct_file_name2 = os.path.join(os.path.join(parent_directory, DATA_DESTINATION_FOLDER_NAME), 
                                    'QC_LAV_Warped_Seg_Masks.npy')
    list_transforms = np.load(npy_ct_file_name1, allow_pickle=True)
    list_warped_segmentations = np.load(npy_ct_file_name2, allow_pickle=True)
    
    # Directory to Save Results
    RESULT_FILE_NAME = 'Results_Est_QC_LAV'
    results_dir = os.path.join(parent_directory, RESULT_FILE_NAME)
    current_version = '1'
    log_save_dir = results_dir + '/' + current_version + '/'
    
    # Run: 
    compute_segmentation_quality()
