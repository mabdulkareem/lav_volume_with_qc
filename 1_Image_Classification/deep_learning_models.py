
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.applications import VGG16, VGG19, ResNet50



def vgg16_model(INPUTS, NO_OUTPUT_LAYER_UNITS, with_last_activation): 
    
    # load the base VGG16 model
    base_model =  VGG16(include_top=False,   # whether to include fully-connected layers at the top of the network.
                        weights=None,        # 'None' (random initialization), or 
                                             # the path to the weights file to be loaded.
                        input_shape=INPUTS)

    # Add a Global Average Pooling Layer
    output = layers.GlobalAveragePooling2D()(base_model.output)

    # Output Layer
    output = layers.Dense(NO_OUTPUT_LAYER_UNITS, activation=with_last_activation)(output)

    # Define the inputs and outputs of the model
    model = Model(base_model.input, output)

    return model
    

def vgg19_model(INPUTS, NO_OUTPUT_LAYER_UNITS, with_last_activation): 
    
    # load the base VGG16 model
    base_model =  VGG19(include_top=False,   # whether to include fully-connected layers at the top of the network.
                        weights=None,        # 'None' (random initialization), or 
                                             # the path to the weights file to be loaded.
                        input_shape=INPUTS)

    # Add a Global Average Pooling Layer
    output = layers.GlobalAveragePooling2D()(base_model.output)

    # Output Layer
    output = layers.Dense(NO_OUTPUT_LAYER_UNITS, activation=with_last_activation)(output)

    # Define the inputs and outputs of the model
    model = Model(base_model.input, output)

    return model

    
def resnet50_model(INPUTS, NO_OUTPUT_LAYER_UNITS, with_last_activation):
    
    # load the base ResNet50 model
    base_model =  ResNet50(include_top=False,   # whether to include fully-connected layers at the top of the network.
                           weights=None,        # 'None' (random initialization), or 
                                                # the path to the weights file to be loaded.
                           input_shape=INPUTS)
    
    # Add a Global Average Pooling Layer
    output = layers.GlobalAveragePooling2D()(base_model.output)

    # Output Layer
    output = layers.Dense(NO_OUTPUT_LAYER_UNITS, activation=with_last_activation)(output)

    # Define the inputs and outputs of the model
    model = Model(base_model.input, output)
    
    return model
