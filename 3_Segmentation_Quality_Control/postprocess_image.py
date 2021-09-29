
import tensorflow as tf



def create_mask(pred_mask):
    '''
    Get the channel with the highest probability to create the segmentation mask. Recall that the output of 
    our UNet has 2 channels. Thus, for each pixel, the predicition will be the channel with the highest probability.
    '''
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    
    return pred_mask
