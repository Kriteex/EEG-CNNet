from metrics_and_plots import accuracy, plot_training_progress, plot_confusion_matrix
from data_preprocess import load_data, save_results
from trainin_and_testin import train, test_all
from model_eegnet import EEGNet, EEGNetV2
import torch
import os


def runs(model, loss_fn, optimizer, people, path, train_losses, test_losses):
    print(f'Loading object {people}\'s data...')
    train_data, train_label, test_data, test_label = load_data(path, people)
    print('Loaded!')
    
    train(model, train_data, train_label, test_data, test_label, loss_fn, optimizer, train_losses, test_losses)


def two_convs(model_name, loss_fn, optimizer, path, save_dir, device, accuracies, conv1_out_channels, conv2_out_channels, conv1_kernel_size, conv2_kernel_size):

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


                    #TODO
                    if optimizer == "Adam":
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
                    else:
                        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

                    train_losses = []
                    test_losses = []

                    for i in range(5):
                        runs(model, loss_fn, optimizer, i, path, train_losses, test_losses)

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

                    test_acc = test_all(path, new_save_dir, model)

                    model_acc_tupple = {folder_name, test_acc}

                    accuracies.append(model_acc_tupple)