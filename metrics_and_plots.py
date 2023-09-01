import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




def accuracy(preds, labels):
    _, preds = torch.max(preds, dim=1)
    return (preds == labels).float().mean()


                
    # Convert lists to tensors
    all_labels = torch.cat(all_labels)
    all_preds = torch.cat(all_preds)

    plot_confusion_matrix(all_labels, all_preds, save_dir)

    print("\n======RESULTS=======\n")
    print("Total accuracy on testing set: " + str(tot_acc.cpu().numpy())[:4])
    
    save_results(str(tot_acc.cpu().numpy())[:4], model, save_dir)







def plot_training_progress(train_losses, test_losses, path):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Training and Test Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(path+"/training_progress.png")
    plt.show()





def plot_confusion_matrix(labels, preds, path, class_names=["Negative","Neutral","Positive"]):
    preds = torch.argmax(preds, dim=1)  # Convert logits to class predictions
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    cm = confusion_matrix(labels, preds)
    
    # Normalize the confusion matrix
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(10, 10))

    # Create a heatmap for the confusion matrix
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
                cbar_kws={'fraction' : 0.01},
                xticklabels=class_names, yticklabels=class_names, ax=ax)

    # Add labels to the plot
    plt.title('Confusion matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(path+"/confusion_matrix.png")
    plt.show()


