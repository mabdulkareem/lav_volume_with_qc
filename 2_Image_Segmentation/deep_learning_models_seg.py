
import tensorflow as tf
#from tensorflow_addons.layers import InstanceNormalization
import numpy as np


# - Two Conv. Layers: 
def downsample_block(filters, size, norm_type='batchnorm', apply_norm=True):
    """Downsamples an input: Conv2D => Batchnorm => LeakyRelu
    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer
    Returns:
        Downsample Sequential Model
    """
    
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()

    # 1. Layer 1
    result.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same', 
                                      kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    # 2. Layer 2
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding='same', 
                                      kernel_initializer=initializer, use_bias=False))

    if apply_norm:
        if norm_type.lower() == 'batchnorm':
            result.add(tf.keras.layers.BatchNormalization())
        elif norm_type.lower() == 'instancenorm':
            result.add(InstanceNormalization())

    result.add(tf.keras.layers.LeakyReLU())

    return result


# - Two Conv. Layers: 
def upsample_block(filters, size, norm_type='batchnorm', apply_dropout=False):
    """Upsamples an input: Conv2DTranspose => Batchnorm => Dropout => Relu
    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_dropout: If True, adds the dropout layer
    Returns:
        Upsample Sequential Model
    """

    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()

    # 1. Layer 1
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                             kernel_initializer=initializer, use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.3))

    result.add(tf.keras.layers.ReLU())

    # 2. Layer 2
    result.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=1, padding='same', 
                                               kernel_initializer=initializer, use_bias=False))

    if norm_type.lower() == 'batchnorm':
        result.add(tf.keras.layers.BatchNormalization())
    elif norm_type.lower() == 'instancenorm':
        result.add(InstanceNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.3))

    result.add(tf.keras.layers.ReLU())

    return result



# Deep Learning Architecture - UNET Architecture
def unet_model(INPUTS, OUTPUT_CHANNELS, last_activation='softmax', norm_type='batchnorm'):
    """DL model 
    Args:
        inputs: Dimension of the input image
        output_channels: Output channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    Returns:
        U-Net model
    """
    
    down_stack = [downsample_block(64, 3, norm_type, apply_norm=False),  
                  downsample_block(128, 3, norm_type),                   
                  downsample_block(256, 3, norm_type),                   
                  downsample_block(512, 3, norm_type),                   
                  downsample_block(1024, 3, norm_type)  
    ]

    up_stack = [  upsample_block(512, 3, norm_type, apply_dropout=True), 
                  upsample_block(256, 3, norm_type, apply_dropout=True),                      
                  upsample_block(128, 3, norm_type, apply_dropout=True),                      
                  upsample_block(64, 3, norm_type)                      
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = tf.keras.layers.Conv2DTranspose( OUTPUT_CHANNELS, 3, strides=2, padding='same',
                                           kernel_initializer=initializer, activation=last_activation)  

    concat = tf.keras.layers.Concatenate()

    inputs = tf.keras.layers.Input(shape=INPUTS)
    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])

    x = last(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=x)
    
    return model
