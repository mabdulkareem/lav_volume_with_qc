
# Import external libraries
import os 
from pathlib import Path
from glob import glob
import shutil
import numpy as np
import nibabel as nib
import cv2
from pystackreg import StackReg


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
        
        
# A function to delete directory
def delete_dir(path_of_dir):
    try: 
        shutil.rmtree(path_of_dir)        
    except FileNotFoundError:
        print('Directory %s or its path doesn\'t exist' % path_of_dir)    
    else: 
        print('Directory %s has been deleted' % path_of_dir)    
        
        
# Create QC transform matrices and warped seg. masks and store in .npy format
def create_qc(): 

    # Create New Data Directories
    path_root, path_dir = os.path.split(DATA_PATH)
    DATA_DESTINATION_PATH = os.path.join(path_root, DATA_DESTINATION_FOLDER_NAME)
    create_dir(DATA_DESTINATION_PATH)

    # Define naming formats: 
    ct_data_nii_gz = '_CT.nii.gz'
    lav_data_nii_gz = '_LAV-label.nii.gz'

    # create list of data files in path
    data_ct_nii_gz_list = glob(os.path.join(DATA_PATH, '*' + ct_data_nii_gz))
    data_lav_nii_gz_list = glob(os.path.join(DATA_PATH, '*' + lav_data_nii_gz))

    # Order using sort
    data_ct_nii_gz_list.sort()
    data_lav_nii_gz_list.sort()

    # Select Data to Use for QC: Build Atlas with Several Patient Data
    no_studies = 5 # No of patients to build an atlas from

    for index in range(no_studies): 
        ct = data_ct_nii_gz_list[index]
        lav = data_lav_nii_gz_list[index]

        # Get the image arrays
        img_arr_ct = nib.load(ct).get_fdata()
        img_arr_lav = nib.load(lav).get_fdata()

        j = 0
        img_shape_lav = img_arr_lav.shape[2]
        par, par_dir = os.path.split(Path(ct))

        for i in range(img_shape_lav): 

            lav_image_i = img_arr_lav[..., i]
            ct_image_i = img_arr_ct[..., i]
            if ((lav_image_i).any() > 0): 

                # Save Both CT and LAV for IMAGE Segmentation Task (i.e. CT Segmentation Mask): 
                # -----------------------------------------------------------------------------
                npy_ct_seg_file_name = os.path.join(DATA_DESTINATION_PATH, 
                                                par_dir.replace(ct_data_nii_gz, '_') + str(f'{j:04}') + '_pixel_array_data.npy')
                np.save(npy_ct_seg_file_name, np.array(ct_image_i))

                npy_lav_seg_file_name = os.path.join(DATA_DESTINATION_PATH, 
                                                par_dir.replace(ct_data_nii_gz, '_') + str(f'{j:04}') + '_seg_data.npy')
                np.save(npy_lav_seg_file_name, np.array(lav_image_i, dtype=np.int32)) # Must only 'integize' segmentation mask image!

            j += 1


    # create list of *npy data files in path
    data_list_img = glob(DATA_DESTINATION_PATH + '/*_pixel_array_data.npy')
    data_list_seg = glob(DATA_DESTINATION_PATH + '/*_seg_data.npy')

    # Order using sort
    data_list_seg.sort()
    data_list_img.sort()

    # Main Program
    list_transforms = []
    list_warped_segmentations = []

    skip = 25
    refslices = [i for i in range(0, len(data_list_img), skip)]
    for refslice_index in refslices: 

        # Load reference ('fixed') image
        img_ct_ref, seg_ct_ref = load_images(DATA_DESTINATION_PATH, refslice_index, data_list_seg, data_list_img)

        for index in refslices:
            if index != refslice_index: 
                print('Index: ', index)

                # Load 'moving' image
                img_ct_moving, seg_ct_moving = load_images(DATA_DESTINATION_PATH, index, data_list_seg, data_list_img)

                # Rigid Body transformation
                sr = StackReg(StackReg.RIGID_BODY)
                reg_image = sr.register_transform(img_ct_ref, img_ct_moving)
                transform_matrix = sr.get_matrix()

                # Compute Warped transform images
                height, width = seg_ct_moving.shape
                warped_seg = cv2.warpPerspective(seg_ct_moving.astype('float32'), 
                                                      transform_matrix, (width, height))

                list_transforms.append(transform_matrix)
                list_warped_segmentations.append(warped_seg)
            else:
                print('skipping slice: ', index)

    print('Length of Transformation/Warped Segmentation Matrices: ', len(list_transforms))

    # Delete all *.npy files and recreate the folder
    delete_dir(DATA_DESTINATION_PATH)
    create_dir(DATA_DESTINATION_PATH)
    
    # Save Transformation Matrix in *.npy
    npy_ct_file_name1 = os.path.join(DATA_DESTINATION_PATH, 'QC_LAV_Transform_Matrix.npy')
    np.save(npy_ct_file_name1, list_transforms)

    # Save Warped Segmentation Masks in *.npy
    npy_ct_file_name2 = os.path.join(DATA_DESTINATION_PATH, 'QC_LAV_Warped_Seg_Masks.npy')
    np.save(npy_ct_file_name2, list_warped_segmentations)
    
    
    
if __name__ == "__main__":
    
    # Define Source Data Directory
    DATA_PATH = 'path/to/dataset/folder'
    
    # Define Destination Directory Name (*.npy will be save in DATA_PATH/DATA_DESTINATION_FOLDER_NAME)
    DATA_DESTINATION_FOLDER_NAME = 'Data_Quality_Control_LAV'
    
    # Create QC transform matrices and warped seg. masks and store in .npy format: 
    create_qc()
