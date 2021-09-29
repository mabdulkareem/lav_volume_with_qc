
import tensorflow as tf
import numpy as np
import os
import cv2
import random
from pathlib import Path
import scipy.ndimage as ndimage

# Data Preparation and Generation Functions and their Definitions and Datasets: 

def prepare_data_list(data_dir, seg_data_npy, pixel_array_data_npy, TEST_RATIO, with_shuffle=False): 
    ''' Given path to data directory, and TEST_RATIO parameter
    Returns: (Lists of path of *.npy files - one for the train and the other for the test)
            - data_list_train
            - data_list_test 
            - data_list_eval'''
    
    
    data_dir = Path(data_dir) 
    
    # List of data sets
    seg_data_list = [str(k) for k in list(data_dir.glob('*' + seg_data_npy))]  # 'Fake' seg_data_list
    
    # Ensure list contains only 'seg_data' with corresponding 'pixel_data'
    pixel_array_data_list = [str(k) for k in list(data_dir.glob('*' + pixel_array_data_npy))]
    seg_data_list_ = []
    for i in range(len(seg_data_list)): 
        seg_data_i = str(seg_data_list[i])
        seg_data_p = seg_data_i.replace(seg_data_npy, pixel_array_data_npy)
        if (seg_data_p in pixel_array_data_list): 
            seg_data_list_.append(seg_data_i)
    
    seg_data_list_.sort()
    seg_data_list = seg_data_list_                     # 'Actual' seg_data_list
    
    if with_shuffle == True: 
        random.shuffle(seg_data_list) # random shuffle list
    else:
        seg_data_list.sort()
        
    # Get no. of image set 
    image_count = len(seg_data_list) 
    
    # Divide data list into 'train', 'test' and 'eval' sets where, for example,
    # if TEST_RATIO = 0.2, then train-test-eval = 80-10-10, and 
    # if TEST_RATIO = 0.15, then train-test-eval = 85-7.5-7.5, and so on. 
    test = int(np.floor(TEST_RATIO/2 * image_count))
    seg_data_list_test = seg_data_list[0:test]
    seg_data_list_eval = seg_data_list[test:2*test]
    seg_data_list_train = seg_data_list[2*test:]
    
    return seg_data_list_train, seg_data_list_test, seg_data_list_eval

def normalize(input_image):
    ''' Given input_image: Normalizes the input_image
    Returns: input_image - Normalized input_image'''
    
    input_image = tf.cast(input_image, tf.float32) / np.max(input_image)
    
    return input_image

# Image Resize with openCV (cv2)
def image_resize(image, IMG_DIM, inter=cv2.INTER_AREA):
    ''' Given an image, resize to width and height''' 
    
    resized_image = cv2.resize(image, IMG_DIM, interpolation=inter)
    
    return resized_image

def get_data(seg_path, IMG_DIM, with_resize=True, with_rotate=True, 
             seg_data_npy='seg_data.npy', pixel_array_data_npy='pixel_array_data.npy'):
    '''Given the path to segmentation mask data, e.g. *.npy, and the desired image dimension
    
    Returns: 
        seg - image data (normalized)
        img_class - class of image e.g. present or absence of tissue  
        
    Assumption: 
        It assumes, for the image classfication problem, images are divided into two 
        folders - one with substring 'ABSENT' ('0' class) and the other with 'PRESENT' ('1' class) depending 
        on the absence or presence of a tissue, respectively. 
    
    Usage Example: img, img_class = get_data(data_list_train[3]) '''    
    
    # Load segmentation mask    
    seg = np.load(seg_path, allow_pickle=True)
    
    # Load image
    img_path = seg_path.replace(seg_data_npy, pixel_array_data_npy)
    img = np.load(img_path, allow_pickle=True)
    
    # Image Resizing
    if with_resize == True:   
        img = image_resize(img, IMG_DIM)
        seg = image_resize(seg.astype('float32'), IMG_DIM) # OpenCV requires conversion to 'float' here
        seg = np.around(seg).astype(int)  # Convert back to integer. 
            
    # Image Normalization
    img = normalize(img)
    
    # Rotate Image (btw -30^o and +30^o)
    if with_rotate == True: 
        rotation_angle = np.random.uniform(-30, 30)
        img = ndimage.rotate(img, rotation_angle, reshape=False)
        seg = ndimage.rotate(seg, rotation_angle, reshape=False)
    
    # Note: TensorFlow requires this additional axis for its batching operation: 
    img = img[:, :, np.newaxis]
    seg = seg[:, :, np.newaxis]
    
    return img, seg
