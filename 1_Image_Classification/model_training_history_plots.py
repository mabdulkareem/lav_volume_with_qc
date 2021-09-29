
import matplotlib.pyplot as plt
import numpy as np

from general_utilities import create_dir

def training_history_plots(model_training_history, save_img_path): 
    
    save_img_path = save_img_path + '/model_training_history_plots'
    create_dir(save_img_path)
    
    acc = model_training_history.history['accuracy']
    loss = model_training_history.history['loss']
    epochs=np.arange(len(acc)) # Get number of epochs

    plt.figure(figsize=(8,8))
    plt.plot(epochs, acc, 'b')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(save_img_path + '/model_training_history_acc_plots.png')
    plt.close()
    
    plt.figure(figsize=(8,8))
    plt.plot(epochs, loss, 'r')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(save_img_path + '/model_training_history_loss_plots.png')
    plt.close()
    
    val_acc = model_training_history.history['val_accuracy']
    val_loss = model_training_history.history['val_loss']
    val_epochs=np.arange(0, len(val_acc)) * np.floor(len(acc)/len(val_acc)) # the last part is for scaling

    plt.figure(figsize=(8,8))
    plt.plot(val_epochs, val_acc, 'b', linestyle="--")
    plt.title('Validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(save_img_path + '/model_training_history_val_acc_plots.png')
    plt.close()
    
    plt.figure(figsize=(8,8))
    plt.plot(val_epochs, val_loss, 'r', linestyle="--")
    plt.title('Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(save_img_path + '/model_training_history_val_loss_plots.png')
    plt.close()
    
    plt.figure(figsize=(8,8))
    plt.plot(epochs, acc, 'b', label='training')
    plt.plot(val_epochs, val_acc, 'b', linestyle="--", label='validation')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig(save_img_path + '/model_training_history_both_acc_plots.png')
    plt.close()
    
    plt.figure(figsize=(8,8))
    plt.plot(epochs, loss, 'b', label='training')
    plt.plot(val_epochs, val_loss, 'b', linestyle="--", label='validation')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig(save_img_path + '/model_training_history_both_loss_plots.png')
    plt.close()
    
