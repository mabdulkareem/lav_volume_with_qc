
import tensorflow as tf
import numpy as np
import cv2


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

def preprocess(img, IMG_DIM, with_resize=True):
    '''Given the path to image data, e.g. */*.npy, and the desired image dimension
    
    Returns: 

        
    Assumption: 
         
    
    Usage Example:    '''
    
    # Image Resizing
    if with_resize == True:         
        img = image_resize(img, IMG_DIM)
        
    # Image Normalization
    img = normalize(img) 
    
    return img[np.newaxis, :, :, np.newaxis]
