import torch
from torch import nn
from typing import Optional


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
                 conv1_kernel_size: int = 4, 
                 conv2_out_channels: int = 62, 
                 conv2_kernel_size: int = 3, 
                 drop_rate: Optional[float] = 0.5):

        super(EEGNet, self).__init__()

        self.convl1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=conv1_out_channels, kernel_size=conv1_kernel_size, stride=1, padding=0),
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


        self.linear_relu_stack1 = nn.Sequential(
            nn.Linear(124, 248),  # adjust input size according to the output size of pooling layer
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

        self.linear_relu_stack2 = nn.Sequential(
            nn.Linear(124, 124),  # adjust input size according to the output size of pooling layer
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