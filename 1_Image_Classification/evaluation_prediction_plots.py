
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn
from sklearn.metrics import confusion_matrix
import seaborn as sns

from general_utilities import create_dir
 
    
# utility to display a row of images with their predictions and true labels
def display_images(image, predictions, labels, n, save_img_path, all_eval=False):
    
    save_img_path = save_img_path + '/model_predictions_using_eval_data'
    if all_eval: 
        save_img_path = save_img_path + '_all'
    create_dir(save_img_path)
        
    image = np.reshape(image, [n, 224, 224])
    for i in np.arange(n): 
        
        plt.figure(figsize=(8,8))
        img = image[i,:,:]
        plt.imshow(tf.keras.preprocessing.image.array_to_img(np.rot90(img[:,:,np.newaxis],3))) # Rotate Nifti (by 90^o three times) to normal way of visualization
        if labels[i] == predictions[i]:
            color = 'green'
        else: 
            color = 'red'
        plt.xlabel('Actual Label: {}; Predicted Label: {}'.format(labels[i], predictions[i]), color=color)
        plt.savefig(save_img_path + '/model_predictions_using_eval_data_' + str('{0:04d}'.format(i)) + '.png')
        plt.close()
        
        

def pred_plots_for_evaluation(class_names, eval_dataset, model, NO_OF_IMAGES, save_img_path, all_eval=False): 
    
    dataset = eval_dataset
    x_batches, y_pred_batches, y_true_batches = [], [], []

    for x, y, _ in dataset:
        y_pred = model(x)
        y_pred_batches = y_pred.numpy()
        y_true_batches = y.numpy()
        x_batches = x.numpy()

    # generate 'size' random numbers in range 0 to 'length of y_pred_batches': 
    indexes = np.random.choice(len(y_pred_batches), size=NO_OF_IMAGES)
    
    images_to_plot = x_batches[indexes]
    y_pred_to_plot = y_pred_batches[indexes]
    y_true_to_plot = y_true_batches[indexes]

    y_pred_labels = [class_names[int(np.around(sel_y_pred.item()))] for sel_y_pred in y_pred_to_plot]
    y_true_labels = []
    for sel_y_true in y_true_to_plot:
        i = np.around(np.asscalar(sel_y_true))
        y_true_labels.append(class_names[i.astype(int)])

    display_images(images_to_plot, y_pred_labels, y_true_labels, NO_OF_IMAGES, save_img_path, all_eval=all_eval)
    
    
    
def obtain_ytrue_ypred(dataset, model, all_eval=False): 
    
    if all_eval: 
        y_pred_batches, y_true_batches = np.array([]), np.array([])
        for x, y, _ in dataset:
            y_pred = model(x)
            y_pred_batches = np.append( y_pred_batches, y_pred.numpy())
            y_true_batches = np.append( y_true_batches, y.numpy())
    else:    
        
        y_pred_batches, y_true_batches = [], []
        for x, y, _ in dataset.take(1):
            y_pred = model(x)
            y_pred_batches = y_pred.numpy()
            y_true_batches = y.numpy()

    y_true_batches_ = [int(np.around(y_true_i.item())) for y_true_i in y_true_batches]
    y_pred_batches_ = [int(np.around(y_pred_i.item())) for y_pred_i in y_pred_batches]

    return y_true_batches_, y_pred_batches_    
    
    
    
def plot_cm(train_dataset, test_dataset, eval_dataset, model, save_img_path): 
    
    '''Confusion Matrix Plots'''
    
    save_img_path = save_img_path + '/model_evaluation_confusion_matrix_plots'
    create_dir(save_img_path)

    train_y_true_batches, train_y_pred_batches = obtain_ytrue_ypred(train_dataset, model)
    test_y_true_batches, test_y_pred_batches = obtain_ytrue_ypred(test_dataset, model)
    eval_y_true_batches, eval_y_pred_batches = obtain_ytrue_ypred(eval_dataset, model)
    
    def plot_and_print(labels, predictions, title): 
        
        print('\n')
        print('Confusion matrix using {} dataset'.format(title))
        
        cm = confusion_matrix(labels, predictions)    
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix using {} dataset'.format(title))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        print('True Negatives: ', tn)
        print('False Positives: ', fp)
        print('False Negatives: ', fn)
        print('True Positives: ', tp)
        
        # Calculate precision
        if (tp + fp == 0):
            precision = 1.0
        else:
            precision = tp / (tp + fp)
        print('Precision: ', precision)
        
        # Calculate recall
        if (tp + fn == 0):
            recall = 1.0
        else:
            recall = tp / (tp + fn)
        print('Recall: ', recall)   
        
        # Calculate F1-Score
        f1_score = 2 * ((precision * recall) / (precision + recall))
        print('F1 Score: ', f1_score)
        
        plt.savefig(save_img_path + '/model_evaluation_confusion_matrix_plots_' + title + '.png')
        plt.close()
    
    plot_and_print(train_y_true_batches, train_y_pred_batches, 'training')
    plot_and_print(test_y_true_batches, test_y_pred_batches, 'testing')
    plot_and_print(eval_y_true_batches, eval_y_pred_batches, 'evaluation') 
    
    
    
def plot_cm_eval(dataset, model, save_img_path, title): 
    
    '''Confusion Matrix Plots'''
    
    save_img_path = save_img_path + '/model_evaluation_confusion_matrix_plots_all'
    create_dir(save_img_path)

    y_true_batches, y_pred_batches = obtain_ytrue_ypred(dataset, model, all_eval=True)
    
    def plot_and_print(labels, predictions, title): 
        
        print('\n')
        print('Confusion matrix using {} dataset'.format(title))
        
        cm = confusion_matrix(labels, predictions)    
        plt.figure(figsize=(5,5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title('Confusion matrix using {} dataset'.format(title))
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        
        tn = cm[0][0]
        fp = cm[0][1]
        fn = cm[1][0]
        tp = cm[1][1]
        print('True Negatives: ', tn)
        print('False Positives: ', fp)
        print('False Negatives: ', fn)
        print('True Positives: ', tp)
        
        # Calculate precision
        if (tp + fp == 0):
            precision = 1.0
        else:
            precision = tp / (tp + fp)
        print('Precision: ', precision)
        
        # Calculate recall
        if (tp + fn == 0):
            recall = 1.0
        else:
            recall = tp / (tp + fn)
        print('Recall: ', recall)   
        
        # Calculate F1-Score
        f1_score = 2 * ((precision * recall) / (precision + recall))
        print('F1 Score: ', f1_score)
        
        plt.savefig(save_img_path + '/model_evaluation_confusion_matrix_plots_' + title + '.png')
        plt.close()
    
    plot_and_print(y_true_batches, y_pred_batches, title)
    
    
    
def plot_roc(train_dataset, test_dataset, eval_dataset, model, save_img_path):
    
    save_img_path = save_img_path + '/model_evaluation_roc_curve_plots'
    create_dir(save_img_path)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def plot_roc_curve(ax, labels, predictions, title, **kwargs): 
    
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

        ax.plot(100*fp, 100*tp, label=title, linewidth=2, **kwargs)
        ax.set_xlabel('False positives [%]')
        ax.set_ylabel('True positives [%]')
        #ax.set_xlim([-0.5,20])
        #ax.set_ylim([80,100.5])
        ax.grid(True)
        ax.set_aspect('equal')
        
        return ax
    
    train_y_true_batches, train_y_pred_batches = obtain_ytrue_ypred(train_dataset, model)
    test_y_true_batches, test_y_pred_batches = obtain_ytrue_ypred(test_dataset, model)
    eval_y_true_batches, eval_y_pred_batches = obtain_ytrue_ypred(eval_dataset, model)
    
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax = plot_roc_curve(ax, train_y_true_batches, train_y_pred_batches, 'training', color=colors[0])
    ax = plot_roc_curve(ax, test_y_true_batches, test_y_pred_batches, 'testing', color=colors[0], linestyle='--')
    ax1 = ax
    ax.legend(loc='lower right')
    ax.figure.savefig(save_img_path + '/model_evaluation_roc_curve_plots_1_train_test.png')
    
    ax1 = plot_roc_curve(ax1, eval_y_true_batches, eval_y_pred_batches, 'evaluation', color=colors[1], linestyle='-.')
    ax1.legend(loc='lower right')
    ax1.figure.savefig(save_img_path + '/model_evaluation_roc_curve_plots_2_train_test_eval.png')
    
    plt.close('all')
    
    

def plot_roc_eval(test_dataset, eval_dataset, model, save_img_path):
    
    save_img_path = save_img_path + '/model_evaluation_roc_curve_plots_all'
    create_dir(save_img_path)
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def plot_roc_curve(ax, labels, predictions, title, **kwargs): 
    
        fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

        ax.plot(100*fp, 100*tp, label=title, linewidth=2, **kwargs)
        ax.set_xlabel('False positives [%]')
        ax.set_ylabel('True positives [%]')
        #ax.set_xlim([-0.5,20])
        #ax.set_ylim([80,100.5])
        ax.grid(True)
        ax.set_aspect('equal')
        
        return ax
    
    test_y_true_batches, test_y_pred_batches = obtain_ytrue_ypred(test_dataset, model, all_eval=True)
    eval_y_true_batches, eval_y_pred_batches = obtain_ytrue_ypred(eval_dataset, model, all_eval=True)
    
    plt.figure(figsize=(5,5))
    ax = plt.gca()
    ax = plot_roc_curve(ax, test_y_true_batches, test_y_pred_batches, 'testing', color=colors[0])
    ax = plot_roc_curve(ax, eval_y_true_batches, eval_y_pred_batches, 'evaluation', color=colors[1], linestyle='--')
    ax.legend(loc='lower right')
    ax.figure.savefig(save_img_path + '/model_evaluation_roc_curve_plots_test_and_eval.png')
    
    plt.close('all')
