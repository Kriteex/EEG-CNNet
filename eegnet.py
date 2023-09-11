from model_eegnet import EEGNet, EEGNetV2
from data_preprocess import load_data, save_results
from metrics_and_plots import accuracy, plot_training_progress, plot_confusion_matrix
from trainin_and_testin import train, test_all, train_losses, test_losses

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



# Initialize the model
model = EEGNet(conv1_out_channels=248)
model = model.to(device)



# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)




def runs(people, path):
    print(f'Loading object {people}\'s data...')
    train_data, train_label, test_data, test_label = load_data(path, people)
    print('Loaded!')
    
    train(model, train_data, train_label, test_data, test_label, loss_fn, optimizer)



if __name__ == '__main__':
    path = '/home/karl/Desktop/EEG/Data/SEED/SEED/SEED_EEG/dependent_data_de_pow_shuffled'
    for i in range(5):
        runs(i, path)

    save_dir = "/home/karl/Desktop/EEG/Base_results"
    plot_training_progress(train_losses, test_losses, save_dir)

    test_all(path, save_dir, model)
