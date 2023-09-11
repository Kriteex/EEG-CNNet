import torch
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler



def load_data(path, people, strategy="UPSAMPLE"):

    smote = SMOTE(sampling_strategy='all')
    undersample = RandomUnderSampler(sampling_strategy='auto')

    train_data = np.load(f'{path}/train_dataset_{people}.npy')
    train_label = np.load(f'{path}/train_labelset_{people}.npy')
    test_data = np.load(f'{path}/test_dataset_{people}.npy')
    test_label = np.load(f'{path}/test_labelset_{people}.npy')

    n_samples, n_channels, n_freq = train_data.shape
    train_data_2d = train_data.reshape((n_samples, n_channels * n_freq))

    n_samples, n_channels, n_freq = test_data.shape
    test_data_2d = test_data.reshape((n_samples, n_channels * n_freq))

    if strategy == "UNDERSAMPLE":
        train_data_under, train_label = undersample.fit_resample(train_data_2d, train_label)
        train_data = train_data_under.reshape((-1, n_channels, n_freq))

        test_data_under, test_label = undersample.fit_resample(test_data_2d, test_label)
        test_data = test_data_under.reshape((-1, n_channels, n_freq))
    elif strategy == "UPSAMPLE":
        train_data_smote, train_label = smote.fit_resample(train_data_2d, train_label)
        train_data = train_data_smote.reshape((-1, n_channels, n_freq))

        test_data_smote, test_label = smote.fit_resample(test_data_2d, test_label)
        test_data = test_data_smote.reshape((-1, n_channels, n_freq))

    return train_data, train_label, test_data, test_label


def save_results(accuracy, model, save_dir):
    # Create a dictionary to store the metrics
    metrics = {
        'Accuracy': accuracy
        #'ROC AUC Macro': roc_macro
    }

    # Convert the dictionary to a pandas DataFrame
    metrics_df = pd.DataFrame(metrics, index=[0])

    # Save the metrics DataFrame to a CSV file
    metrics_df.to_csv(os.path.join(save_dir, 'metrics.csv'), index=False)

    # Save the model
    torch.save(model.state_dict(), os.path.join(save_dir, 'model.pt'))