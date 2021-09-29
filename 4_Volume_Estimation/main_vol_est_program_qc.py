

# Import General Packages: 
import os
import time
from pathlib import Path
from glob import glob

import numpy as np
import tensorflow as tf
import pandas as pd
import nibabel as nib

import warnings 
warnings.filterwarnings("ignore")


# Import User-defined Modules: 
from device_settings import gpu_memory_limit
from general_utilities import create_dir
from preprocess_image import preprocess
from postprocess_image import create_mask
from dice_estimation import estimate_dice


def compute_volume():
    # Device Settings: 
    # Limit memory of GPU/CPU to use 
    memory_limit = 18432          
    gpu_memory_limit(memory_limit)
    
    # Parameters: 
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
    NO_SEG_SKIP = 8     # Due to computational time, we compute dice only for every NO_SEG_SKIP'th segmentation. 
    TOL_IGNORE = 1e-3   # Sets predicted segmentation dice score to ignore when computing quality control score. 
                        # Generally, TOL_IGNORE less are attempt by segmentation model to predict a slice that
                        # it shouldn't have contoured in the first places (i.e. misses of identification model).
    THRESH_QC = 0.70    # Threshold QC value
    
    # Data List Preparation: 
    # ----------------------    
    print('Starting to prepare patient data list...')

    # Use pandas to read output measurement csv file
    df_measurement_table = pd.read_csv(os.path.join(DATA_PATH, measurements_csv))
    df_measurement_table.set_index("patient_id", inplace = True)
    
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
    print('Volume estimation has started.')
    
    current_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print('Volume Estimation Start Time: ', current_time)
    
    # Start Counting time for training
    start = time.perf_counter()
    
    patient_list = []
    label_volume_list = []
    pred_volume_list = []
    qc_score_mean_list = []
    qc_score_perc_list = [] # Percentage of seg. qc score over THRESH_QC (arbitrary) threshold in a list
    qc_score_perc_no_list = [] # No. of seg. qc score above THRESH_QC (arbitrary) threshold in a list (i.e. numerator for computing percentage)
    
    for ct in data_ct_nii_gz_list: 
        
        loaded_img = nib.load(ct)
        img_array_ct = loaded_img.get_fdata()

        patient = os.path.split(ct)[-1] # Get patient file name
        patient = patient.replace(ct_data_nii_gz, '') # Get patient name: Remove the prefix "_CT.nii.gz"
        
        print('- - - - - - - - - - - - - - - - - - - - -')
        print('Patient: ', patient)
        
        # Read nifti header data: 
        # -----------------------
        file_header = loaded_img.header
        pixdim = file_header['pixdim'][1] # Returns x pix dimension (note: pixdim_x = pixdim_y)
        pixdim_z = file_header['pixdim'][3] # Returns z pix dimension 
        
        # Read output measurement data: 
        # -----------------------------
        try: 
            actual_volume = df_measurement_table.loc[patient][parameter]
        except KeyError:
            print('Patient ', patient, ' not found on measurement table.')
            print('', flush=True)
            continue
            
        print('\n', flush=True)
        
        array_vol_est = []
        pred_dice_list = []
        j = 0
        img_shape_ct = img_array_ct.shape[2]
        for i in range(img_shape_ct): 
                        
            ct_image_i = img_array_ct[..., i]
            ct_image_i = preprocess(ct_image_i, IMG_DIM)
            y_pred = model_ident(ct_image_i)
            y_pred = y_pred.numpy()
            
            # Convert numpy array element to a standard Python scalar, and then round: 
            y_pred = int(np.around(y_pred.item()))
            
            if y_pred == 1: # i.e. if slide needs to be contoured. 
                
                y_pred_seg = model_seg(ct_image_i)    # Size: [1, 224, 224, 2]
                y_pred_seg = create_mask(y_pred_seg)  # Size: [1, 224, 224, 1]
                
                array_vol_est.append(y_pred_seg[0, ..., 0])  #Store results
                
                # Estimate Dice
                if j%NO_SEG_SKIP == 0: # Due to computational cost, compute dice for the (no_seg_skip)'th slice
                    
                    pred_dice = estimate_dice(y_pred_seg[0, ..., 0], 
                                              list_transforms, 
                                              list_warped_segmentations, 
                                              IMG_DIM)
                
                    pred_dice_list.append(pred_dice)  #Store results
                
                j += 1
              
        array_vol_est = np.asarray(array_vol_est) # Convert the list to an array
        
        const = 5e-3 # Distance between slices is given by 'pixdim_z * const' (i.e. pixdim_z is direction vector)
                      #(Note: not in NifTi - Obtained using "actual_volume/(vol_sum*pixdim_z)")
                      # where 'vol_sum' was computed by multiplying by 'const'. 

        vol_sum = 0
        no_img_slices = array_vol_est.shape[0]
        for i in range(no_img_slices): 
            
            img_1 = array_vol_est[i, ...]
            if i < (no_img_slices-1): 
                img_2 = array_vol_est[i+1, ...]
            else:  # Handles the case of last slice. 
                img_2 = img_1
                
            no_pixel_1 = np.count_nonzero(img_1 > 0) 
            no_pixel_2 = np.count_nonzero(img_2 > 0) 
            
            # Method
            A_i = no_pixel_1 * pixdim **2
            A_i_1 = no_pixel_2 * pixdim **2
            vol_i = (A_i + A_i_1)/2 
            
            vol_sum += vol_i   
        
        vol_sum *= (pixdim_z * const) # i.e. vol_sum = vol_sum * dist. btw slices
        
        # Print computed volume
        print('Computed Vol: ', vol_sum)
        
        # Print the actual volume
        print('Actual Vol: ', actual_volume)  
        
        # Others
        print('x-y Pix Dim: ', pixdim)
        print('z Pix Dim: ', pixdim_z)
        
        # Estimate Quality Control (QC) Score
        pred_dice_list = [i for i in pred_dice_list if i>TOL_IGNORE] # ignore irrelevant values
        try: 
            qc_score_mean = np.mean(pred_dice_list) # mean of the list
            len_total = len(pred_dice_list)
            len_thresh = len([i for i in pred_dice_list if i>THRESH_QC])
            qc_score_perc = len_thresh/len_total
            qc_score_perc_no = len_thresh
        except:   # In case 'pred_dice_list' is empty, prevents division-by-zero
            qc_score_mean = 0 
            qc_score_perc = 0
            qc_score_perc_no = 0
            
        print('QC Score MEAN: ', qc_score_mean)
        print('QC List PERC: ', qc_score_perc)
        print('QC List PERC. No.: ', qc_score_perc_no)
        
        #Store results 
        patient_list.append(patient)
        label_volume_list.append(actual_volume)
        pred_volume_list.append(vol_sum)
        qc_score_mean_list.append(qc_score_mean)
        qc_score_perc_list.append(qc_score_perc)
        qc_score_perc_no_list.append(qc_score_perc_no)
        
        print('\n', flush=True)
     
    print('- - - - - - - - - - - - - - - - - - - - -')
    
    # Store times
    elapsed = time.perf_counter() - start
    print('Volume Estimation Elapsed Time: %.3f seconds.' % elapsed)
    
    end_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print('Volume Estimation Finish Time: ', end_time)
        
    # Store dataframe results in dataframe: |Patient|Actual Volume|Predicted Volume|
    df_table = pd.DataFrame({'Patient': patient_list, 
                             'Label_Volume': label_volume_list, 
                             'Pred_Volume': pred_volume_list, 
                             'QC_Score_mean': qc_score_mean_list,
                             'QC_Score_perc': qc_score_perc_list, 
                             'QC_Score_perc_no': qc_score_perc_no_list
                            })
        
    data_file_name = log_save_dir + 'results'
    df_table.to_csv(data_file_name + '.csv', encoding='utf-8', index=True)
    df_table.to_excel(data_file_name + '.xlsx', index=True) 
    
    print('\n', flush=True)
    
    
        
if __name__ == "__main__":
    
    # LAV VOLUME
    # ----------
    parent_directory = 'path/to/dataset/parent-directory/folder'
    
    # CT Image Data Path: 
    DATA_DIR_NAME = 'folder-name'
    DATA_PATH = os.path.join(parent_directory, DATA_DIR_NAME)
    
    # Path to Classification Model: 
    ident_model = 'path/to/model-classification-folder/model_in_h5_format.h5'
    MODEL_IDENT_PATH = os.path.join(parent_directory, ident_model)
    
    # Path to Slide Segmentation Model: 
    seg_model = 'path/to/model-segmentation-folder/model_in_h5_format.h5'
    MODEL_SEG_PATH = os.path.join(parent_directory, seg_model)
    
    # Define Path and Load Quality Control Transformation Matrix: 
    npy_ct_file_name1 = os.path.join(os.path.join(parent_directory,'Data_Quality_Control_LAV'), 
                                    'QC_LAV_Transform_Matrix.npy')
    npy_ct_file_name2 = os.path.join(os.path.join(parent_directory,'Data_Quality_Control_LAV'), 
                                    'QC_LAV_Warped_Seg_Masks.npy')
    list_transforms = np.load(npy_ct_file_name1, allow_pickle=True)
    list_warped_segmentations = np.load(npy_ct_file_name2, allow_pickle=True)
    
    # Directory to Save Results
    RESULT_FILE_NAME = 'Results_Est_Vol_LAV_withQC'
    results_dir = os.path.join(parent_directory, RESULT_FILE_NAME)
    current_version = '1'
    log_save_dir = results_dir + '/' + current_version + '/'
    
    # Define naming formats: 
    measurements_csv = 'data_output_variable_table.csv'
    
    # Run: 
    parameter = 'LAV_Vol'
    compute_volume()
