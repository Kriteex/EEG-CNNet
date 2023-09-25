from model_eegnet import EEGNet, EEGNetV2
from data_preprocess import load_data, save_results
from metrics_and_plots import accuracy, plot_training_progress, plot_confusion_matrix
from trainin_and_testin import train, test_all

import os
import torch
import numpy as np
import pandas as pd
from torch import nn
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
path = '/home/karl/Desktop/EEG/Data/SEED/SEED/SEED_EEG/dependent_data_de_pow_shuffled'
model_name = "EEGNetV2"

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()

if model_name == "EEGNet":
    conv1_out_channels = [8, 16, 32, 64, 128, 258]
    conv1_kernel_size = [1,2,3,4,5]
    conv2_out_channels = [8, 16, 32, 64, 128, 258]
    conv2_kernel_size = [1,2,3,4,5]
elif model_name == "EEGNetV2":
    conv1_out_channels = [32, 64, 128]
    conv1_kernel_size = [3,5,7,13,25,50]
    conv2_out_channels = [32, 64, 128]
    conv2_kernel_size = [3,5,7,13,25,50]


def runs(people, path, train_losses, test_losses):
    print(f'Loading object {people}\'s data...')
    train_data, train_label, test_data, test_label = load_data(path, people)
    print('Loaded!')
    
    train(model, train_data, train_label, test_data, test_label, loss_fn, optimizer, train_losses, test_losses)



if __name__ == '__main__':
    save_dir = "/home/karl/Desktop/EEG/EEG-CNNet/Base_results/"

    for conv1_ch in conv1_out_channels:
        for conv2_ch in conv2_out_channels:
            for ker1 in conv1_kernel_size:
                for ker2 in conv2_kernel_size:

                    # Initialize the model
                    if model_name == "EEGNet":
                        model = EEGNet(conv1_out_channels=conv1_ch, conv1_kernel_size=ker1, conv2_out_channels=conv2_ch, conv2_kernel_size=ker2)
                    elif model_name == "EEGNetV2":
                        model = EEGNetV2(conv1_out_channels=conv1_ch, conv1_kernel_size=ker1, conv2_out_channels=conv2_ch, conv2_kernel_size=ker2)

                    
                    
                    model = model.to(device)

                    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

                    train_losses = []
                    test_losses = []

                    for i in range(5):
                        runs(i, path, train_losses, test_losses)

                    # Create a folder name based on the value of 'i'
                    folder_name = f"folder_{conv1_ch}_{ker1}_{conv2_ch}_{ker2}"

                    new_save_dir = save_dir + folder_name
                    
                    # Check if the folder already exists
                    if not os.path.exists(new_save_dir):
                        # If it doesn't exist, create it
                        os.mkdir(new_save_dir)
                        print(f"Created folder: {folder_name}")
                    else:
                        print(f"Folder {folder_name} already exists.")

                    plot_training_progress(train_losses, test_losses, new_save_dir)

                    test_all(path, new_save_dir, model)

