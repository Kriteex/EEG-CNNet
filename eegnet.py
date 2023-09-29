from model_eegnet import EEGNet, EEGNetV2
from data_preprocess import load_data, save_results
from metrics_and_plots import accuracy, plot_training_progress, plot_confusion_matrix
from trainin_and_testin import train, test_all
from architecture_search_grid_loops import two_convs

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
optimizer = "Adam"

if model_name == "EEGNet":
    conv1_out_channels = [8]#, 16, 32, 64, 128, 258]
    conv1_kernel_size = [1]#,2,3,4,5]
    conv2_out_channels = [8]#, 16, 32, 64, 128, 258]
    conv2_kernel_size = [1]#,2,3,4,5]
elif model_name == "EEGNetV2":
    conv1_out_channels = [32]#, 64, 128]
    conv1_kernel_size = [3]#,5,7,13,25,50]
    conv2_out_channels = [32]#, 64, 128]
    conv2_kernel_size = [3]#,5,7,13,25,50]



if __name__ == '__main__':
    save_dir = "/home/karl/Desktop/EEG/EEG-CNNet/Base_results/"

    accuracies = []
    two_convs(model_name, loss_fn, optimizer, path, save_dir, device, accuracies, conv1_out_channels, conv2_out_channels, conv1_kernel_size, conv2_kernel_size)   

    accuracies = pd.DataFrame(accuracies, columns=["accuracy", "model"])
    accuracies.to_csv("accuracies.csv")


