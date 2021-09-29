
# Import General Packages: 
import numpy as np
import os
import tensorflow as tf
import warnings 
warnings.filterwarnings("ignore")
import time


# Import User-defined Modules: 
from dataset_generation import prepare_data_list, get_data
from device_settings import gpu_memory_limit
from evaluation_prediction_plots import pred_plots_for_evaluation, plot_cm, plot_cm_eval, plot_roc, plot_roc_eval



def evaluate_model(supply_data_list=None, supply_model=None, supply_others=None):
    
    global data_dir, results_dir, current_version, MODEL_NAME
    
    # Device Settings: Limit memory of GPU to use 
    if supply_data_list is None: 
        memory_limit = 16384 
        gpu_memory_limit(memory_limit)
    
    # Parameters: 
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    IMG_DIM = (IMG_WIDTH, IMG_HEIGHT)
    BATCH_SIZE = 128                       # The number of consecutive elements of a dataset to combine in a single batch.
    TEST_RATIO = 0.15                      # Define Training-Testing(Validating)-Evaluation Dataset Ratio 
                                           # where '0.2' implies '0.8-0.1-0.1', '0.15' implies '0.85-0.075-0.075', and so on. 
        
    if supply_others is not None: 
        data_dir, results_dir, current_version, MODEL_NAME = supply_others
        
    path_name = results_dir + 'model'                                             # Directory for storing model 
    model_name = 'model_' + MODEL_NAME + '/' + current_version                    # Name of new model 
    MODEL_PATH_NAME = path_name + '/' + model_name
    
    # Data List Preparation: 
    # ----------------------    
    print('Starting to prepare data list...')
    
    if supply_data_list is None: 
        data_list_train, data_list_test, data_list_eval = prepare_data_list(data_dir, TEST_RATIO)
    else: 
        data_list_train, data_list_test, data_list_eval = supply_data_list
    
    TRAIN_LENGTH = len(data_list_train)
    TEST_LENGTH = len(data_list_test)
    EVAL_LENGTH = len(data_list_eval)
    
    print('No. of data points in training dataset: {}'.format(TRAIN_LENGTH))
    print('No. of data points in testing dataset: {}'.format(TEST_LENGTH))
    print('No. of data points in evaluation dataset: {}'.format(EVAL_LENGTH))
    
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
            img_new, label = get_data(i, IMG_DIM, with_rotate=False)
            
            # Sample weight
            if label[0] == 0: 
                sample_weight = np.array([ CLASS_WEIGHTS[0] ])
            elif label[0] == 1: 
                sample_weight = np.array([ CLASS_WEIGHTS[1] ])
                
            sample_weight = sample_weight[:, np.newaxis]
            
            yield img_new, label, sample_weight            
            
    def data_generator_test():
        for i in data_list_test:
            img_new, label = get_data(i, IMG_DIM, with_rotate=False)
            
            # Sample weight
            if label[0] == 0: 
                sample_weight = np.array([ CLASS_WEIGHTS[0] ])
            elif label[0] == 1: 
                sample_weight = np.array([ CLASS_WEIGHTS[1] ])
                
            sample_weight = sample_weight[:, np.newaxis]
            
            yield img_new, label, sample_weight
            
    def data_generator_eval():
        for i in data_list_eval:
            img_new, label = get_data(i, IMG_DIM, with_rotate=False)
            
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
    def batch_dataset(ds, BATCH_SIZE):
        ds = ds.batch(BATCH_SIZE) 
        return ds
    
    train_dataset = batch_dataset(dataset_train, BATCH_SIZE)
    test_dataset = batch_dataset(dataset_test, BATCH_SIZE)   
    eval_dataset = batch_dataset(dataset_eval, BATCH_SIZE)
    
    # Define number of 'repeat()' steps: 
    no_of_train_batches = np.ceil(TRAIN_LENGTH/BATCH_SIZE)
    no_of_test_batches = np.ceil(TEST_LENGTH/BATCH_SIZE)
    no_of_eval_batches = np.ceil(EVAL_LENGTH/BATCH_SIZE)
    
    print('Datasets generation completed.  \n')
    # ----------- End of Datasets generation ------------- 


    # Load Model: 
    # ---------------
    print('Loading model...')

    model_path_name =  MODEL_PATH_NAME + '/model_in_h5_format.h5'
        
    # Recreate the exact same model, including weights and optimizer.
    if supply_model is None: 
        #model = tf.keras.models.load_model(model_path_name)
        model = tf.keras.models.load_model(model_path_name,compile=False)
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0001),
                      loss=tf.keras.losses.BinaryCrossentropy(name='loss'))
    else: 
        model = supply_model
    
    print('Model loading completed. \n')
    # ----------- End of Model Loading -------------
    
    
    # Model Evaluation: 
    # -----------------    
    print('Model evaluation begins...')
    
    # Model Evaluation on Unseen Dataset (a batch from dataset_eval (i.e. eval_dataset))
    # ----------------------------------------------------------------------------------
    start = time.perf_counter() # Start counting time for this computation
    eval_results = model.evaluate(eval_dataset, verbose=0,
                                  steps=no_of_eval_batches) # 'steps': no. of iterations to cover all data in dataset
    print('Model Evaluation Using Unseen Dataset: - loss, - accuracy:', eval_results)
    elapsed = time.perf_counter() - start
    print('Model Evaluation Elapsed Time: %.3f seconds.' % elapsed)    
    
    # Visualisation of Model Predictions from Evaluation Dataset
    # ----------------------------------------------------------
    NO_OF_IMAGES = 200 # No of images to visualise; this must be smaller than the BATCH_SIZE 'times' no_of_eval_batches
    class_names = [0, 1]
    
    eval_dataset_repeat_once = eval_dataset.repeat(1) # Note: To avoid infinite loop, it is crucial to set this to a fixed number. 
                                                      # 'repeat(1)' here ensures batches cover all data in datasets once. 
    eval_dataset_repeat_once = eval_dataset_repeat_once.take(no_of_eval_batches) # Take all batches
    pred_plots_for_evaluation(class_names, eval_dataset_repeat_once, model, NO_OF_IMAGES, MODEL_PATH_NAME, all_eval=True)

    # Plot the confusion matrices (cm) and evaluate model for Precision, Recall, and F1 Score. 
    # ----------------------------------------------------------------------------------------
    start = time.perf_counter() # Start counting time for this computation
    eval_dataset_all = eval_dataset.repeat(1).take(no_of_eval_batches)
    plot_cm_eval(eval_dataset_all, model, MODEL_PATH_NAME, 'eval')
    elapsed = time.perf_counter() - start
    print('Evaluation Data - Time taken to compute the confusion matrix: %.3f seconds.' % elapsed)
    
    start = time.perf_counter() # Start counting time for this computation
    test_dataset_all = test_dataset.repeat(1).take(no_of_test_batches)
    plot_cm_eval(test_dataset_all, model, MODEL_PATH_NAME, 'test')
    elapsed = time.perf_counter() - start
    print('Test Data - Time taken to compute the confusion matrix: %.3f seconds.' % elapsed)
    
    # Plot the roc_curve: 
    # -------------------
    start = time.perf_counter() # Start counting time for this computation
    plot_roc_eval(test_dataset_all, eval_dataset_all, model, MODEL_PATH_NAME)
    elapsed = time.perf_counter() - start
    print('Time taken to compute the ROC curve: %.3f seconds.' % elapsed)

    print('Model evaluation completed. \n')
    # ----------- End of Model Evaluation -------------
    
if __name__ == "__main__":
    
    global data_dir, results_dir, current_version, MODEL_NAME  
    
    # LAV IDENTIFICATION
    # ----------------------
    data_dir = 'path/to/dataset/folder'
    results_dir = 'path/to/folder/to/save/results/'
    current_version = '1'
    
    # Model Architecture 1: 
    #MODEL_NAME = 'VGG16'
    #evaluate_model()
    
    # Model Architecture 2: 
    #MODEL_NAME = 'VGG19'
    #evaluate_model()
    
    # Model Architecture 3: 
    MODEL_NAME = 'ResNet'
    evaluate_model()
