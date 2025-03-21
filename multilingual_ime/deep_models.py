"""
This module defines deep learning models for language detection and token detection tasks.
"""

from torch import nn


class LanguageDetectorModel(nn.Module):
    """
    A neural network model for language detection.
    This model is implemented using PyTorch's `nn.Module` and consists of a 
    sequential stack of fully connected layers with ReLU activations, dropout 
    for regularization, and a final softmax layer for classification.
    Attributes:
        model (nn.Sequential): The sequential model containing the layers of 
            the neural network.
    Args:
        input_shape (int): The size of the input features.
        num_classes (int): The number of output classes for classification.
    Methods:
        forward(x):
            Performs a forward pass through the model.
            Args:
                x (torch.Tensor): The input tensor.
            Returns:
                torch.Tensor: The output tensor after passing through the model.
    """

    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
            nn.Softmax(dim=0),
        )

    def forward(self, x):
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to be passed through the model.
        Returns:
            torch.Tensor: Output tensor after processing by the model.
        """
        return self.model(x)


class TokenDetectorModel(nn.Module):
    """
    TokenDetectorModel is a neural network model designed for token detection tasks. 
    It utilizes a feedforward architecture with multiple fully connected layers and 
    activation functions to process input data and produce predictions.
    Attributes:
        model (nn.Sequential): A sequential container of layers including linear 
            transformations, ReLU activations, and a Sigmoid activation at the end 
            for producing output probabilities.
    Methods:
        __init__(input_shape, num_classes):
            Initializes the TokenDetectorModel with the specified input shape and 
            number of output classes.
        forward(x):
            Performs a forward pass through the model to compute the output tensor 
            based on the input tensor.
    """
    def __init__(self, input_shape, num_classes):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.Linear(32, num_classes),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Perform a forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor to be passed through the model.
        Returns:
            torch.Tensor: Output tensor after being processed by the model.
        """
        return self.model(x)
