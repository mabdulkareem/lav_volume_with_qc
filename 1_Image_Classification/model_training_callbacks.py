
import os 
import tensorflow as tf
import numpy as np
import gc


def call_callbacks(LOG_DIR, LEARNING_RATE, MODEL_PATH_NAME, VAL_FREQ, 
                   TEST_LENGTH, dataset_test, train_dataset, test_dataset, model): 
    
    par_dir, _ = os.path.split(LOG_DIR)
    
    # Callback 1. Learning rate callback -----------------------------
    learning_rate_log_dir = par_dir + '/learning_rate_log'
    learning_rate_summary_writer = tf.summary.create_file_writer(learning_rate_log_dir)

    def lr_schedule(epoch):
        """
        Returns a custom learning rate that decreases as epochs progress.
        """
        learning_rate = LEARNING_RATE
        
        # VGG16/19 and ResNet Training for 60 epochs with LEARNING_RATE = 1e-4
        if epoch > 10:
            learning_rate = 0.5 * LEARNING_RATE
        if epoch > 15:
            learning_rate = 0.2 * LEARNING_RATE
        if epoch > 20:
            learning_rate = 0.1 * LEARNING_RATE
        if epoch > 25:
            learning_rate = 0.05 * LEARNING_RATE
        if epoch > 30:
            learning_rate = 0.02 * LEARNING_RATE
        if epoch > 35:
            learning_rate = 0.01 * LEARNING_RATE
        if epoch > 40:
            learning_rate = 0.005 * LEARNING_RATE
        if epoch > 45:
            learning_rate = 0.002 * LEARNING_RATE
        if epoch > 50:
            learning_rate = 0.001 * LEARNING_RATE

        # Log the learning_rate data.
        with learning_rate_summary_writer.as_default():
            tf.summary.scalar('learning_rate', data=learning_rate, step=epoch)

        return learning_rate

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
    # ----------------------- END of Callback 1. ----------------------
    
    # Callback 2. Create a callback that saves the model's weights every 'VAL_FREQ' steps ---------
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(   filepath=MODEL_PATH_NAME,
                                                                verbose=0, 
                                                                save_freq=VAL_FREQ, # an integer
                                                                monitor='val_loss', # 'val_loss' or 'loss'
                                                                mode='auto' ) 
    # ----------------------- END of Callback 2. --------------------------------------------------
    
    # Callback 3. Interrupt training if `val_loss` stops improving for over specified 'patience'. 
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(patience=8*VAL_FREQ, monitor='loss', mode='min', min_delta=1e-4)
    # ----------------------- END of Callback 3. ------------------------------------------------------
    
    # Callback 4. Garbage collection to prevent memory leak: Forcing garbage collection at every epoch
    # Define the per-epoch callback.
    def garbage_collection(epoch, logs):

        tf.keras.backend.clear_session()        
        gc.collect()
        
    garbage_collection_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=garbage_collection)
    # ----------------------- END of Callback 4. ------------------------------------------------------
    
    
    callbacks = [
                 lr_callback,               # callback 1
                 checkpoint_callback,       # callback 2
                 early_stopping_callback,   # callback 3  
                 garbage_collection_callback # callback 4
    ]
    
    return callbacks
