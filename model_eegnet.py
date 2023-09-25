import torch
import numpy as np
from torch import nn
from math import *
from typing import Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class EEGNet(nn.Module):
    """
    EEGNet model for EEG signal classification.

        Initialize the EEGNet model.

        Parameters:
        - in_channels (int): Number of input channels. Default is 62.
        - conv1_out_channels (int): Number of output channels for the first convolution layer. Default is 124.
        - conv1_kernel_size (int): Kernel size for the first convolution layer. Default is 3.
        - conv2_out_channels (int): Number of output channels for the second convolution layer. Default is 62.
        - conv2_kernel_size (int): Kernel size for the second convolution layer. Default is 3.
        - drop_rate (float, optional): Dropout rate. Default is 0.5.
    """
    def __init__(self, 
                 in_channels: int = 62, 
                 conv1_out_channels: int = 124, 
                 conv1_kernel_size: int = 3, 
                 conv2_out_channels: int = 124, 
                 conv2_kernel_size: int = 3, 
                 drop_rate: Optional[float] = 0.5):

        super(EEGNet, self).__init__()

        self.convl1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
            )
        self.convl2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
            )

        # Calculate output shape after Conv1D
        O_conv1 = floor((5 - conv1_kernel_size + 2) / 1) + 1
        
        # Calculate output shape after MaxPool1D
        O_maxpool1 = floor((O_conv1 - 2) / 2) + 1
        
        # Calculate output shape after Flatten
        calculated_O_flatten1 = O_maxpool1 * conv1_out_channels


        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(calculated_O_flatten1, 124),  # adjust input size according to the output size of pooling layer
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 124),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 62),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(62, 3),
            nn.Softmax(dim=1)
        )

        # Calculate output shape after Conv1D
        O_conv2 = floor((5 - conv2_kernel_size + 2) / 1) + 1
        
        # Calculate output shape after MaxPool1D
        O_maxpool2 = floor((O_conv2 - 2) / 2) + 1
        
        # Calculate output shape after Flatten
        calculated_O_flatten2 = O_maxpool2 * conv2_out_channels

        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(calculated_O_flatten2, 124),  # adjust input size according to the output size of pooling layer
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 124),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 62),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(62, 3),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.convl1(x)
        x2 = self.convl2(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        logits1 = self.linear_relu_stack1(x1)
        logits2 = self.linear_relu_stack2(x2)
        averaged_logits = torch.stack([logits1, logits2]).mean(dim=0)
        return averaged_logits


class EEGNetV2(nn.Module):
    """
    EEGNetV2 model for EEG signal classification.

        Initialize the EEGNet model.

        Parameters:
        - in_channels (int): Number of input channels. Default is 62.
        - conv1_out_channels (int): Number of output channels for the first convolution layer. Default is 124.
        - conv1_kernel_size (int): Kernel size for the first convolution layer. Default is 3.
        - conv2_out_channels (int): Number of output channels for the second convolution layer. Default is 62.
        - conv2_kernel_size (int): Kernel size for the second convolution layer. Default is 3.
        - drop_rate (float, optional): Dropout rate. Default is 0.5.
    """
    def __init__(self, 
                 in_channels: int = 5, 
                 conv1_out_channels: int = 20, 
                 conv1_kernel_size: int = 3, 
                 conv2_out_channels: int = 20, 
                 conv2_kernel_size: int = 3, 
                 drop_rate: Optional[float] = 0.5):

        super(EEGNetV2, self).__init__()

        self.convl1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
            )
        self.convl2 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv2_out_channels, kernel_size=conv2_kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Flatten()
            )

        # Calculate output shape after Conv1D
        O_conv1 = floor((62 - conv1_kernel_size + 2) / 1) + 1
        
        # Calculate output shape after MaxPool1D
        O_maxpool1 = floor((O_conv1 - 2) / 2) + 1
        
        # Calculate output shape after Flatten
        calculated_O_flatten1 = O_maxpool1 * conv1_out_channels



        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(calculated_O_flatten1, 248),  # adjust input size according to the output size of pooling layer
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(248, 124),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 62),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(62, 3),
            nn.Softmax(dim=1)
        )

        # Calculate output shape after Conv1D
        O_conv2 = floor((62 - conv2_kernel_size + 2) / 1) + 1
        
        # Calculate output shape after MaxPool1D
        O_maxpool2 = floor((O_conv2 - 2) / 2) + 1
        
        # Calculate output shape after Flatten
        calculated_O_flatten2 = O_maxpool2 * conv2_out_channels

        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(calculated_O_flatten2, 124),  # adjust input size according to the output size of pooling layer
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 124),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(124, 62),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(62, 3),
            nn.Softmax(dim=1)
        )
        self.dropout = nn.Dropout(p=drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = np.transpose(x.cpu(), (0, 2, 1)).to(device)

        x1 = self.convl1(x)
        x2 = self.convl2(x)
        x1 = self.dropout(x1)
        x2 = self.dropout(x2)
        logits1 = self.linear_relu_stack1(x1)
        logits2 = self.linear_relu_stack2(x2)
        averaged_logits = torch.stack([logits1, logits2]).mean(dim=0)
        return averaged_logits