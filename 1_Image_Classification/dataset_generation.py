
import tensorflow as tf
import numpy as np
import os
import cv2
import random
from pathlib import Path
import scipy.ndimage as ndimage

# Data Preparation and Generation Functions and their Definitions and Datasets: 

def prepare_data_list(data_dir, TEST_RATIO, with_shuffle=True): 
    ''' Given path to data directory, and TEST_RATIO parameter
    Returns: (Lists of path of *.npy files - one for the train and the other for the test)
            - data_list_test
            - data_list_train '''
    
    
    data_dir = Path(data_dir) 

    # List of data sets
    data_list = [str(k) for k in list(data_dir.glob('*/*.npy'))]  
    if with_shuffle == True: 
        random.shuffle(data_list) # random shuffle list
    else:
        data_list.sort()
        
    # Get no. of image set 
    image_count = len(data_list) 

    # Divide data list into 'train', 'test' and 'eval' sets where, for example,
    # if TEST_RATIO = 0.2, then train-test-eval = 80-10-10, and 
    # if TEST_RATIO = 0.15, then train-test-eval = 85-7.5-7.5, and so on.  
    test = int(np.floor(TEST_RATIO/2 * image_count))
    data_list_test = data_list[0:test]
    data_list_eval = data_list[test:2*test]
    data_list_train = data_list[2*test:]
    
    return data_list_train, data_list_test, data_list_eval

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

def get_data(img_path, IMG_DIM, with_resize=True, with_rotate=True):
    '''Given the path to image data, e.g. */*.npy, and the desired image dimension
    
    Returns: 
        img - image data (normalized)
        img_class - class of image e.g. present or absence of tissue  
        
    Assumption: 
        It assumes, for the image classfication problem, images are divided into two 
        folders - one with substring 'ABSENT' ('0' class) and the other with 'PRESENT' ('1' class) depending 
        on the absence or presence of a tissue, respectively. 
    
    Usage Example: img, img_class = get_data(data_list_train[3]) '''    
    
    # Load and resize the image    
    img = np.load(img_path, allow_pickle=True)
    
    # Image Resizing
    if with_resize == True:         
        img = image_resize(img, IMG_DIM)
        
    # Image Normalization
    img = normalize(img) 
    
    # Rotate Image (btw -10^o and +10^o)
    if with_rotate == True: 
        img = ndimage.rotate(img, np.random.uniform(-10, 10), reshape=False)
        
    # Obtain label
    label_tag = tf.strings.split(img_path, os.sep)[-2]
    if 'ABSENT' in str(label_tag.numpy()): 
        label = np.array([0])
    elif 'PRESENT' in str(label_tag.numpy()):
        label = np.array([1])
    
    # Note: TensorFlow requires this additional axis for its batching operation: 
    img = img[:, :, np.newaxis]
    label = label[:, np.newaxis]
    
    return img, label
