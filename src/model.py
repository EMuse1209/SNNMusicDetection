"""
Spiking Neural Network model for speech recognition.
"""

import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, surrogate
from spikingjelly.activation_based.functional import reset_net

from .config import TAU, V_THRESHOLD, V_RESET, DETACH_RESET, ALPHA

class SNN(nn.Module):
    """Spiking Neural Network for speech recognition.
    
    This model uses a convolutional architecture with Leaky Integrate-and-Fire (LIF)
    neurons for speech command recognition.
    """
    
    def __init__(self, num_classes: int = 12):
        """Initialize the SNN model.
        
        Args:
            num_classes: Number of output classes (default: 12 for speech commands)
        """
        super().__init__()
        
        # Convolutional layers with LIF neurons
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(4, 3), stride=(1, 1), padding=(2, 1), bias=False),
            neuron.LIFWrapper(
                neuron.LIFNode(
                    v_threshold=V_THRESHOLD,
                    v_reset=V_RESET,
                    detach_reset=DETACH_RESET,
                    step_mode='m',
                    tau=TAU,
                    surrogate_function=surrogate.Sigmoid(alpha=ALPHA)
                )
            ),
            nn.Conv2d(64, 64, kernel_size=(4, 3), stride=(1, 1), padding=(6, 3), dilation=(4, 3), bias=False),
            neuron.LIFWrapper(
                neuron.LIFNode(
                    v_threshold=V_THRESHOLD,
                    v_reset=V_RESET,
                    detach_reset=DETACH_RESET,
                    step_mode='m',
                    tau=TAU,
                    surrogate_function=surrogate.Sigmoid(alpha=ALPHA)
                )
            ),
            nn.Conv2d(64, 64, kernel_size=(4, 3), stride=(1, 1), padding=(24, 9), dilation=(16, 9), bias=False),
            neuron.LIFWrapper(
                neuron.LIFNode(
                    v_threshold=V_THRESHOLD,
                    v_reset=V_RESET,
                    detach_reset=DETACH_RESET,
                    step_mode='m',
                    tau=TAU,
                    surrogate_function=surrogate.Sigmoid(alpha=ALPHA)
                )
            ),
        )
        
        # Fully connected layer
        self.fc = nn.Linear(2560, num_classes, bias=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x: Input tensor of shape (batch_size, 1, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        # Reset the network state
        reset_net(self)
        
        # Forward pass through convolutional layers
        x = self.conv(x)
        
        # Flatten the output
        x = x.view(x.size(0), -1)
        
        # Forward pass through fully connected layer
        x = self.fc(x)
        
        return x 