
import tensorflow as tf

def gpu_memory_limit(memory_limit): 
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 'memory_limit'. A GPU machine may have several 'logical GPU's'
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit)])   # Logical GPU 1
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            print('GPU memory limit allocated.')
            
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
