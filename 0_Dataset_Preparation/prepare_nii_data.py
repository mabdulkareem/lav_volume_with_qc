
# Import external libraries
import os 
from pathlib import Path
from glob import glob
import numpy as np
import nibabel as nib

# A function to create directory
def create_dir(path_of_dir):
    try:
        os.makedirs(path_of_dir) # For one directory containing inner/sub directory(ies)    
    except FileExistsError:
        print("Directory %s already exists" % path_of_dir)   
    except OSError:
        print ("Creation of the directory %s failed" % path_of_dir)     
    else:
        print ("Successfully created the directory %s " % path_of_dir)   
    
    
def prepare_data():
    
    # Create New Data Directories
    DATA_DESTINATION_FOLDER_NAME = 'Data_Ident_LAV'
    SUB_DIR_1 = 'LAV_ABSENT'
    SUB_DIR_2 = 'LAV_PRESENT'
    DATA_DESTINATION_FOLDER_SEG_LAV = 'Data_Seg_LAV'

    path_root, path_dir = os.path.split(DATA_PATH)

    DATA_DESTINATION_PATH = os.path.join(path_root, DATA_DESTINATION_FOLDER_NAME)
    DATA_DESTINATION_PATH_SUBDIR_1 = os.path.join(DATA_DESTINATION_PATH, SUB_DIR_1)
    DATA_DESTINATION_PATH_SUBDIR_2 = os.path.join(DATA_DESTINATION_PATH, SUB_DIR_2)
    DATA_DESTINATION_PATH_SEG_LAV = os.path.join(path_root, DATA_DESTINATION_FOLDER_SEG_LAV)

    create_dir(DATA_DESTINATION_PATH)
    create_dir(DATA_DESTINATION_PATH_SUBDIR_1)
    create_dir(DATA_DESTINATION_PATH_SUBDIR_2)
    create_dir(DATA_DESTINATION_PATH_SEG_LAV)

    # Define naming formats: 
    ct_data_nii_gz = '_CT.nii.gz'
    lav_data_nii_gz = '_LAV-label.nii.gz'

    # create list of data files in path
    data_ct_nii_gz_list = glob(os.path.join(DATA_PATH, '*' + ct_data_nii_gz))
    data_lav_nii_gz_list = glob(os.path.join(DATA_PATH, '*' + lav_data_nii_gz))

    # Order using sort
    data_ct_nii_gz_list.sort()
    data_lav_nii_gz_list.sort()

    # Print length of each list
    print('No. of CT nii files {} and no. of LAV nii files {}'.format(len(data_ct_nii_gz_list), len(data_lav_nii_gz_list)))

    # Main Task: Datasets for Tissue Identification and Segmentation Tasks for LAV
    k = 0
    for ct, lav in zip(data_ct_nii_gz_list, data_lav_nii_gz_list): 
        if k >= 0: 
            print('-----')
            patient_i = Path(ct)
            par, par_dir = os.path.split(patient_i)
            print('Working on {} ...'.format(par_dir))  

            # Exta Error Check: Ensure CT and LAV match (e.g. Abl001_CT is matched with Abl001_LAV)
            data_ct_nii_gz_checked_list = data_ct_nii_gz_list
            data_lav_nii_gz_checked_list = []

            for i in range(len(data_ct_nii_gz_checked_list)): 
                ct_data_i = str(data_ct_nii_gz_checked_list[i])
                lav_data_j = ct_data_i.replace(ct_data_nii_gz, lav_data_nii_gz)
                if lav_data_j in data_lav_nii_gz_list: 
                    data_lav_nii_gz_checked_list.append(lav_data_j)

            data_lav_nii_gz_checked_list.sort()

            assert len(data_ct_nii_gz_checked_list) == len(data_lav_nii_gz_checked_list), "Something is wrong with the CT and LAV lists."

            img_arr_ct = nib.load(ct).get_fdata()
            img_arr_lav = nib.load(lav).get_fdata()

            if img_arr_ct.shape[2] != img_arr_lav.shape[2]: 
                print('Skipping {} ...'.format(par_dir)) 
                continue

            j = 0
            img_shape_lav = img_arr_lav.shape[2]

            if classification_data: 
                for i in range(img_shape_lav): 

                    lav_image_i = img_arr_lav[..., i]
                    ct_image_i = img_arr_ct[..., i]

                    # For Classification: 
                    if ((lav_image_i).any() > 0): # Ideally, same as: 'if ((laf_image_i).any() > 0)'

                        # Save CT in Class B :- SUB_DIR_2 = 'LAV_PRESENT'
                        npy_ct_file_name = os.path.join(DATA_DESTINATION_PATH_SUBDIR_2, 
                                                        par_dir.replace(ct_data_nii_gz, '_') + str(f'{j:04}') + '_pixel_array_data.npy')
                        np.save(npy_ct_file_name, np.array(ct_image_i))

                    else: 

                        # Save CT in Class A :- SUB_DIR_1 = 'LAV_ABSENT'
                        npy_ct_file_name = os.path.join(DATA_DESTINATION_PATH_SUBDIR_1, 
                                                        par_dir.replace(ct_data_nii_gz, '_') + str(f'{j:04}') + '_pixel_array_data.npy')
                        np.save(npy_ct_file_name, np.array(ct_image_i))

                    j += 1

            if segmentation_data: 
                for i in range(img_shape_lav): 

                    lav_image_i = img_arr_lav[..., i]
                    ct_image_i = img_arr_ct[..., i]

                    # For Segmentation: 
                    if ((lav_image_i).any() > 0): # Ideally, same as: 'if ((laf_image_i).any() > 0)'

                        # Save Both CT and LAV for IMAGE Segmentation Task (i.e. CT Segmentation Mask): 
                        # -----------------------------------------------------------------------------
                        npy_ct_seg_file_name = os.path.join(DATA_DESTINATION_PATH_SEG_LAV, 
                                                        par_dir.replace(ct_data_nii_gz, '_') + str(f'{j:04}') + '_pixel_array_data.npy')
                        np.save(npy_ct_seg_file_name, np.array(ct_image_i))

                        npy_lav_seg_file_name = os.path.join(DATA_DESTINATION_PATH_SEG_LAV, 
                                                        par_dir.replace(ct_data_nii_gz, '_') + str(f'{j:04}') + '_seg_data.npy')
                        np.save(npy_lav_seg_file_name, np.array(lav_image_i, dtype=np.int32)) # Must only 'integize' segmentation mask image!

                    j += 1

        if k == work_with: 
            break
        else: 
            k += 1

    if classification_data: 
        # Count the number of data in classification task
        data_ct_absent_list = glob(DATA_DESTINATION_PATH_SUBDIR_1 + '/*.npy')
        data_ct_present_list = glob(DATA_DESTINATION_PATH_SUBDIR_2 + '/*.npy')
        print('No of "absent" class images: ', len(data_ct_absent_list))
        print('No of "present" class images: ', len(data_ct_present_list))

    if segmentation_data:
        # Count the number of data in LAV segmentation task
        data_ct_seg_mask_list_lav_1 = glob(DATA_DESTINATION_PATH_SEG_LAV + '/*_seg_data.npy')
        data_ct_seg_mask_list_lav_2 = glob(DATA_DESTINATION_PATH_SEG_LAV + '/*_pixel_array_data.npy')
        data_ct_seg_mask_list_lav_1.sort()
        data_ct_seg_mask_list_lav_2.sort()
        print('No of ct and segmentation mask pairs for LAV: ', len(data_ct_seg_mask_list_lav_1), ', ', len(data_ct_seg_mask_list_lav_2))


        
if __name__ == "__main__":  

    
    # Source Data Path - Directory Containing Dataset:
    DATA_PATH = '/path/to/folder'
    
    # LAV CLASSIFICATION/SEGMENTATION
    # -------------------------------
    work_with = 150 # - For example, work with 150 out of 337 datasets
    classification_data = False # Must be changed to True if segmentation_data is False
    segmentation_data = True # Must be changed to True if classfication_data is False
    prepare_data()
