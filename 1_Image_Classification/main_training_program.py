
# Import General Packages: 
import numpy as np
import os
import tensorflow as tf
import warnings 
warnings.filterwarnings("ignore")
import time
from functools import partial


# Import User-defined Modules: 
from dataset_generation import prepare_data_list, get_data
from deep_learning_models import vgg16_model, vgg19_model, resnet50_model
from device_settings import gpu_memory_limit
from model_training_callbacks import call_callbacks
from evaluation_prediction_plots import pred_plots_for_evaluation, plot_cm, plot_roc
from model_training_history_plots import training_history_plots
from main_evaluation_program import evaluate_model


def train_model():
    
    # Device Settings: 
    # Limit memory of GPU to use 
    memory_limit = 16384          
    gpu_memory_limit(memory_limit)
    
    # Parameters: 
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
    BATCH_SIZE = 128                       # The number of consecutive elements of a dataset to combine in a single batch.
    BUFFER_SIZE = 2048                     # 2048 = '2 GB' memory for buffering
    TEST_RATIO = 0.15                      # Define Training-Testing(Validating)-Evaluation Dataset Ratio 
                                           # where '0.2' implies '0.8-0.1-0.1', '0.15' implies '0.85-0.075-0.075', and so on. 
    EPOCHS = 60
    LEARNING_RATE = 1e-4
    VAL_FREQ = 3                           # 'validation_freq' - specifies no. of training epochs to run before 
                                           # a new validation run is performed 
        
    path_name = results_dir + 'model'                                             # Directory for storing model 
    model_name = 'model_' + MODEL_NAME + '/' + current_version                    # Name of new model 
    MODEL_PATH_NAME = path_name + '/' + model_name
    
    LOG_BASE_DIR = results_dir + "logs"                             
    LOG_DIR = LOG_BASE_DIR + '/' + model_name + "/fit/"             # Logging directory for tensorboard
    
    # Clear any logs from previous runs
    if CLEAR_LOG: 
        cmd = "rm -rf " + LOG_DIR
        os.system(cmd)
        os.makedirs(LOG_DIR, exist_ok=True)
    
    # Data List Preparation: 
    # ----------------------    
    print('Starting to prepare data list...')
    
    data_list_train, data_list_test, data_list_eval = prepare_data_list(data_dir, TEST_RATIO)
    
    print('No. of data points in training dataset: {}'.format(len(data_list_train)))
    print('No. of data points in testing dataset: {}'.format(len(data_list_test)))
    print('No. of data points in evaluation dataset: {}'.format(len(data_list_eval)))
    
    print('Data list preparation completed. \n')
    # ---- End of Datasets List Preparation ----
    
    
    # Deal With Data Imbalance: 
    # -------------------------
    # An approach to avoid bias by placing more importance on one class over another during training. 
    # The 'less the no. in a class', the more 'weight' the class has. 
    # Formula: class_weight_of_class_i = (1/no_of_items_in_class_i) * (no_of_items_in_total/no_of_classes)
    CLASS_0_List = []
    CLASS_1_List = []
    for img_path in data_list_train: 
        label_tag = tf.strings.split(img_path, os.sep)[-2]
        if 'ABSENT' in str(label_tag.numpy()): 
            CLASS_0_List.append(np.array([0]))
        elif 'PRESENT' in str(label_tag.numpy()):
            CLASS_1_List.append(np.array([1]))
    NO_IN_CLASS_0 = len(CLASS_0_List)
    NO_IN_CLASS_1 = len(CLASS_1_List)
    print('NO_IN_CLASS_0: ', NO_IN_CLASS_0)
    print('NO_IN_CLASS_1: ', NO_IN_CLASS_1)
    TOTAL = NO_IN_CLASS_0 + NO_IN_CLASS_1
    NO_OF_CLASSES =  2.0
    WEIGHT_FOR_0 = (1 / NO_IN_CLASS_0)*(TOTAL)/NO_OF_CLASSES 
    WEIGHT_FOR_1 = (1 / NO_IN_CLASS_1)*(TOTAL)/NO_OF_CLASSES

    CLASS_WEIGHTS = {0: WEIGHT_FOR_0, 1: WEIGHT_FOR_1}  
    
    print('Data imbalance addressed through weighting: {}'.format(CLASS_WEIGHTS))
    # ----------- End of Dealing With Data Imbalance ------------------------------- 
    
    
    # Datasets (training and testing/validation datasets): 
    # ----------------------------------------------------
    print('Starting to generate datasets...')
    # Dataset Generators: For the 'train' and 'test' sets
    def data_generator_train():
        for i in data_list_train:
            img_new, label = get_data(i, IMG_DIM)
            
            # Sample weight
            if label[0] == 0: 
                sample_weight = np.array([ CLASS_WEIGHTS[0] ])
            elif label[0] == 1: 
                sample_weight = np.array([ CLASS_WEIGHTS[1] ])
                
            sample_weight = sample_weight[:, np.newaxis]
            
            yield img_new, label, sample_weight            
            
    def data_generator_test():
        for i in data_list_test:
            img_new, label = get_data(i, IMG_DIM)
            
            # Sample weight
            if label[0] == 0: 
                sample_weight = np.array([ CLASS_WEIGHTS[0] ])
            elif label[0] == 1: 
                sample_weight = np.array([ CLASS_WEIGHTS[1] ])
                
            sample_weight = sample_weight[:, np.newaxis]
            
            yield img_new, label, sample_weight
            
    def data_generator_eval():
        for i in data_list_eval:
            img_new, label = get_data(i, IMG_DIM)
            
            # Sample weight
            if label[0] == 0: 
                sample_weight = np.array([ CLASS_WEIGHTS[0] ])
            elif label[0] == 1: 
                sample_weight = np.array([ CLASS_WEIGHTS[1] ])
                
            sample_weight = sample_weight[:, np.newaxis]
            
            yield img_new, label, sample_weight
            
    # Datasets: 
    # 1. Training Dataset
    dataset_train = tf.data.Dataset.from_generator(data_generator_train,
                                                   output_types=(tf.float32, tf.float32, tf.float32), 
                                                   output_shapes=([IMG_WIDTH, IMG_HEIGHT, 1], [1, 1], [1, 1]))
    # 2. Validation Dataset (Used During Training)
    dataset_test = tf.data.Dataset.from_generator(data_generator_test,
                                                   output_types=(tf.float32, tf.float32, tf.float32), 
                                                   output_shapes=([IMG_WIDTH, IMG_HEIGHT, 1], [1, 1], [1, 1])) 
    # 3. Model Evaluation Dataset (Not Used During Training)
    dataset_eval = tf.data.Dataset.from_generator(data_generator_eval,
                                                   output_types=(tf.float32, tf.float32, tf.float32), 
                                                   output_shapes=([IMG_WIDTH, IMG_HEIGHT, 1], [1, 1], [1, 1]))
    # Batching of Dataset: 
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    def configure_for_performance(ds):
        ds = ds.shuffle(buffer_size=BUFFER_SIZE) # Has no effect when steps_per_epoch is not None.
        ds = ds.repeat() # NB: repeat() before batch() seems BETTER than batch() before repeat; see - https://www.tensorflow.org/guide/data
        ds = ds.batch(BATCH_SIZE) 
        #ds = ds.cache()
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    
    train_dataset = configure_for_performance(dataset_train)
    test_dataset = configure_for_performance(dataset_test)
    # No need to use 'configure_for_performance' function: 
    #test_dataset = dataset_test.batch(BATCH_SIZE)    
    eval_dataset = dataset_eval.batch(BATCH_SIZE)
    
    print('Datasets generation completed.  \n')
    # ----------- End of Datasets generation -------------
    
    
    # More Parameters: 
    TRAIN_LENGTH = len(data_list_train)
    TEST_LENGTH = len(data_list_test)
    EVAL_LENGTH = len(data_list_eval)
    
    TRAIN_SUBSPLITS = 10 # Subsplitting to enable us use only portion of the dataset in each epoch
    STEPS_PER_EPOCH = np.floor(TRAIN_LENGTH/BATCH_SIZE/TRAIN_SUBSPLITS)     # No. of batch iterations before a 
                                                                            # training epoch is considered finished
    VAL_SUBSPLITS = 5 # Subsplitting to enable us use only portion of the dataset in each epoch
    VALIDATION_STEPS = np.floor(TEST_LENGTH/BATCH_SIZE/VAL_SUBSPLITS)     # VALIDATION_STEPS is similar to STEPS_PER_EPOCH 
                                                                          # except that it is for validation dataset  
    # Ensure STEPS_PER_EPOCH and VALIDATION_STEPS are at least 1.
    if STEPS_PER_EPOCH < 1: 
        STEPS_PER_EPOCH = 1
    if VALIDATION_STEPS < 1: 
        VALIDATION_STEPS = 1
    

    # Model Creation: 
    # ---------------
    print('Creating new model.')

    # Specify 'NUMBER_OF_CLASSES' output unit(s) and activation function of the last layer
    INPUTS = (IMG_WIDTH, IMG_HEIGHT, 1)
    NO_OUTPUT_LAYER_UNITS = 1
    with_last_activation = 'sigmoid'

    # Model 1: VGG16 Network
    if MODEL_NAME == 'VGG16': 
        model = vgg16_model(INPUTS, NO_OUTPUT_LAYER_UNITS, with_last_activation)            
            
    # Model 2: VGG19 Network
    if MODEL_NAME == 'VGG19': 
        model = vgg19_model(INPUTS, NO_OUTPUT_LAYER_UNITS, with_last_activation)        
        
    # Model 3: ResNet
    if MODEL_NAME == 'ResNet': 
        model = resnet50_model(INPUTS, NO_OUTPUT_LAYER_UNITS, with_last_activation)
        
    # ----------- End of Model Creation -------------
    
    # Model Compilation: 
    # ------------------

    print('Starting to compile model...')
    
    # Define metrics
    METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy', dtype=tf.float32, threshold=0.5)]
    
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
                  loss=tf.keras.losses.BinaryCrossentropy(name='loss'),
                  metrics=METRICS)

    # Model Details 
    model.summary()
    tf.keras.utils.plot_model(model, to_file=results_dir + MODEL_NAME + '_Ver_' + current_version + '_model.png', show_shapes=True ) 
    
    print('Model compilation completed. \n')
    # ----------- End of Model Compilation ---------------    
    
    
    # CALLBACK ISSUES: 
    # ----------------
    print('Preparing TensorFlow callbacks...')
    
    callbacks = call_callbacks(LOG_DIR, LEARNING_RATE, MODEL_PATH_NAME, VAL_FREQ, 
                               TEST_LENGTH, dataset_test, train_dataset, test_dataset, model)
    
    print('TensorFlow callbacks preparation completed.')
    # --------- End of CALLBACK ISSUES -------------------    
    
    
    # Train Model: 
    # ------------
    print('Model training has started...')
    
    # Start Counting time for training
    start = time.perf_counter()
    
    current_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print('Model Training Start Time: ', current_time)
    
    try: 
        model_history = model.fit(train_dataset, 
                                  epochs=EPOCHS,
                                  steps_per_epoch=STEPS_PER_EPOCH,
                                  validation_steps=VALIDATION_STEPS, 
                                  validation_data=test_dataset, 
                                  validation_freq=VAL_FREQ,
                                  callbacks=callbacks)

    except KeyboardInterrupt:
        print("\n KeyboardInterrupt has been caught. Model training has stopped and model is being saved. \n")
        model.save(MODEL_PATH_NAME + '/model_in_h5_format.h5')
        print('Model Saved! \n')
        exit(0)

    elapsed = time.perf_counter() - start
    print('Model Training Elapsed Time: %.3f seconds.' % elapsed)
    
    end_time = time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())
    print('Model Training Finish Time: ', end_time)
        
    model.save(MODEL_PATH_NAME + '/model_in_h5_format.h5')
    print('Model optimization completed. \n')
    # ----------- End of Model Training -------------

    
    # Model Training Plots:
    # ---------------------
    print('Model training history plots started...')
    
    #print('Model History Keys: ', model_history.history.keys())
    print('Model Metric Names: ', model.metrics_names)
    
    training_history_plots(model_history, MODEL_PATH_NAME)
    
    print('Model training history plots. \n')
    # ----------- End of Model Training Plots -------    
    
    
    # Model Evaluation: 
    # -----------------    
    print('Model evaluation begins...')
    
    # Model Evaluation on Unseen Dataset (a batch from dataset_eval (i.e. eval_dataset))
    eval_results = model.evaluate(eval_dataset, verbose=0)
    print('Model Evaluation Using Unseen Dataset: - loss, - accuracy:', eval_results)
    
    # 1. Batched Dataset
    # ------------------
    # Visualisation of Model Predictions from Evaluation Dataset
    NO_OF_IMAGES = 50 # No of images to visualise; these could be repeated has we generate 50 random no. of specific range (e.g. range(10))
    class_names = [0, 1]
    pred_plots_for_evaluation(class_names, eval_dataset, model, NO_OF_IMAGES, MODEL_PATH_NAME)
    
    # Plot the confusion matrices (cm) and evaluate model for Precision, Recall, and F1 Score. 
    plot_cm(train_dataset, test_dataset, eval_dataset, model, MODEL_PATH_NAME)
 
    # Plot the roc_curve
    plot_roc(train_dataset, test_dataset, eval_dataset, model, MODEL_PATH_NAME)
    
    # 2. Complete Dataset
    # -------------------
    supply_data_list = data_list_train, data_list_test, data_list_eval
    supply_others = data_dir, results_dir, current_version, MODEL_NAME
    evaluate_model(supply_data_list=supply_data_list, supply_model=model, supply_others=supply_others)
    
    print('Model evaluation completed. \n')
    # ----------- End of Model Evaluation -------------
    
    
    
if __name__ == "__main__":  
    
    # LAV IDENTIFICATION
    # ----------------------
    data_dir = 'path/to/dataset/folder'
    results_dir = 'path/to/folder/to/save/results/'
    current_version = '1'
    CLEAR_LOG = True
    
    # Model Architecture 1: 
    #MODEL_NAME = 'VGG16'
    #train_model()
    
    # Model Architecture 2: 
    #MODEL_NAME = 'VGG19'
    #train_model()
    
    # Model Architecture 3: 
    MODEL_NAME = 'ResNet'
    train_model()
    
