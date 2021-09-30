Atrial Volume Estimation with Quality Control for Cardiac CT Images using Deep Learning
============

These codes are the outcome of the following paper: M. Abdulkareem  et. al., 'Atrial Volume Estimation with Quality Control for Cardiac CT Images using Deep Learning', 2021.  

The structure of this repository is: 

    ├── LICENSE
    ├── README.md          
    ├── 0_Dataset_Preparation
    ├── 1_Image_Classification
    ├── 2_Image_Segmentation
    ├── 3_Segmentation_Quality_Control
    ├── 4_Volume_Estimation

For easy replicaition of the results of the paper, each directory is self-contained and performs a specific function. 



1. Directory "0_Dataset_Preparation": This contains a data preparation setup code. The entry point is: 

cd ../0_Dataset_Preparation\
python prepare_nii_data.py

It assumes that the user has defined the path to the dataset directory (DATA_PATH = '/path/to/folder'). The assumed structure of the dataset directory is: 

    ├── dataset_directory
    │   ├── Patient_0001_CT.nii.gz      
    │   ├── Patient_0001_LAV-label.nii.gz        
    │   ├── Patient_0002_CT.nii.gz      
    │   ├── Patient_0002_LAV-label.nii.gz 
    │   ├── ...   
    │   ├── Patient_000N_CT.nii.gz      
    │   ├── Patient_000N_LAV-label.nii.gz

That is, the directory contains CT images and their corresponding segmentation mask, both in nifti format (\*.nii.gz) for each patient. Depending on whether we want to perform image classification (classification_data = True, segmentation_data = False) or image segmentation task (classification_data = False, segmentation_data = True), the program creates: 

For classification task, a folder "Data_Ident_LAV" (containing two directories) with the following structure: 

    ├── Data_Ident_LAV
    │   ├── PRESENT      
    │   │   └── Patient_0001_*_pixel_array_data.npy    
    │   │   └── Patient_0001_*_pixel_array_data.npy 
    │   │   └── ... 
    │   │   └── Patient_000N_*_pixel_array_data.npy
    │   │   
    │   ├── ABSENT      
    │   │   └── Patient_0002_*_pixel_array_data.npy    
    │   │   └── Patient_0002_*_pixel_array_data.npy
    │   │   └── ... 
    
The PRESENT and ABSENT folder contain \*.nii.gz converted into numpy arrays for each slice and saved, depending on whether or not the slice contains the left atrium (LA). 

For segmentation task, a folder "Data_Seg_LAV" with the following structure: 

    ── Data_Seg_LAV      
    ├──  Patient_0001_*_pixel_array_data.npy    
    ├──  Patient_0001_*_seg_data.npy
    ├──  Patient_0002_*_pixel_array_data.npy    
    ├──  Patient_0002_*_seg_data.npy
    ├──  ... 
    ├──  Patient_000N_*_pixel_array_data.npy    
    ├──  Patient_000N_*_seg_data.npy

The \*\_pixel_array_data.npy and \*\_seg_data.npy are converted numpy array files for each CT slice containing LA and its corresponding segmentation mask. 



2. Directory "1_Image_Classification": This contains python codes for training VGG16, VGG19 and ResNet50 classification models as described in the paper. The entry point is: 

cd ../1_Image_Classification\
python main_training_program.py

It assumes that the user has defined the path to the dataset directory (data_dir = 'path/to/dataset/folder' - i.e. path to the "Data_Ident_LAV" directory created already) and the location to save the model results (results_dir = 'path/to/folder/to/save/results/').



3. Directory "2_Image_Segmentation": This contains python codes for training the UNet segmentation model as described in the paper. The entry point is: 

cd ../2_Image_Segmentation\
python main_training_program_seg.py

It assumes that the user has defined the path to the dataset directory (data_dir = 'path/to/dataset/folder' - i.e. path to the "Data_Seg_LAV" directory created already) and the location to save the model results (results_dir = 'path/to/folder/to/save/results/').



4. Directory "3_Segmentation_Quality_Control": This contains python codes for obtaining the rigid body transformation and warped segmentation masks as described in the paper. The entry point is: 

cd ../3_Segmentation_Quality_Control\
python main_seg_qc_program.py

It assumes that the user has defined the path to the dataset directory (DATA_PATH = 'path/to/dataset/folder' directory containing nifti files \*_CT.nii.gz and \*_LAV-label.nii.gz as in 1. above).

Note: The results of this are the numpy arrays of the rigid body transformation and warped segmentation masks saved in the folder DATA_DESTINATION_FOLDER_NAME = 'Data_Quality_Control_LAV' in the parent direcotry of DATA_PATH.

It also contains the program for predicting the dice score for segmentation masks in the absence of the ground truths as described in the paper. The program can be run using:  

cd ../3_Segmentation_Quality_Control\
python main_qc_est_program.py

It assumes that the user has defined the paths to the following directories: 

a. Path to the parent directory (parent_directory = 'path/to/dataset/parent-directory/folder'). 

b. The name of the directory within the parent directory in (a.) above containing 'folder-name' containing \*\_CT.nii.gz files whose image slices are to be segmented. 

c. Path to the classification model (e.g. ResNet model - 'path/to/model-classification-folder/model_in_h5_format.h5') to classify and identify images containing LA. 

d. Path to the segmentation model (e.g. UNet model - 'path/to/model-segmentation-folder/model_in_h5_format.h5') to create segmentation masks for images containing LA.

Notes: 
i. Assumes the parent directory (a. above) contains the directory DATA_DESTINATION_FOLDER_NAME = 'Data_Quality_Control_LAV' where the numpy arrays of the rigid body transformation and warped segmentation masks were saved. \
ii. The results of this program are table files (\*.csv and \*.xlsx) containing to columns namely, label (actual) dice score 'Label_Seg_Quality' and the predicted dice score 'Pred_Seg_Quality'. The content of the two files are the same except that one is a csv file and the other is an Excel file. These two files are saved in a newly created folder 'Results_Est_QC_LAV' in the parent directory. 



5. Directory "4_Volume_Estimation":  This contains python codes for obtaining the LA volume (LAV) as described in the paper. The entry point is: 

cd ../4_Volume_Estimation\
python main_vol_est_program_qc.py

It assumes that the user has defined the paths to the following directories: 

a. - d. Same as paths defined in (a. - d.) in (4.) above. 

Notes: 
i. Assumes the parent directory (a. above) contains the file measurements_csv = 'data_output_variable_table.csv' - a csv file that contains the ground truths with two columns namely, patient_id (name of patient e.g. Patient_0001, Patient_0002, etc.) and LAV_Vol (LAV ground truth). \
ii. The results of this program are table files (\*.csv and \*.xlsx) containing to columns namely: 

'Patient', \
'Label_Volume', \
'Pred_Volume', \
'QC_Score_mean', \
'QC_Score_perc', and \
'QC_Score_perc_no'.

The content of the two files are the same except that one is a csv file and the other is an Excel file. These two files are saved in a newly created folder 'Results_Est_Vol_LAV_withQC' in the parent directory. 

The directory "4_Volume_Estimation" also contains the program for flagging LAV results that may be inaccurate as described in the paper. The program can be run using: 

cd ../4_Volume_Estimation\
python main_vol_est_program_qc_flagged_cases.py

That creates a new file 'results_flagged.csv' in the 'Results_Est_Vol_LAV_withQC' directory containing the list of flagged cases. 

