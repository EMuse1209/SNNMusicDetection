"""
Training functions for the SNN speech recognition model.
"""

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

from spikingjelly.activation_based.functional import reset_net

from .config import (
    LEARNING_RATE, NUM_EPOCHS, WARMUP_EPOCHS, GAMMA,
    LABEL_DICT
)
from .model import Net

def test_single_sample(net, test_dataset, device):
    """
    Tests the neural network on a random audio sample from the test dataset.

    Args:
        net: The trained neural network model
        test_dataset: The test dataset containing audio samples
        device: The device (CPU/GPU) to run the model on

    Returns:
        None
        Displays the test results including:
        - Sample index being tested
        - Input tensor shape
        - Actual and predicted labels
        - Prediction confidence
        - Probability distribution plot
    """
    # Select random sample
    sample_idx = torch.randint(0, len(test_dataset), (1,)).item()
    audio, label = test_dataset[sample_idx]

    # Get label mapping
    label_names = list(LABEL_DICT.keys())
    actual_label = label_names[label]

    # Run inference
    net.eval()
    with torch.no_grad():
        # Prepare input tensor
        audio = audio.squeeze()
        audio = audio.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, time, features]
        audio = audio.to(device)

        # Log sample info
        print(f"Testing sample {sample_idx}")
        print(f"Input shape: {audio.shape}")

        # Get model prediction
        output = net(audio)
        predicted_label_idx = output.argmax(dim=1).item()
        predicted_label = label_names[predicted_label_idx]

        # Calculate prediction probabilities
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]

        # Display results
        print(f"Actual label: {actual_label}")
        print(f"Predicted label: {predicted_label}")
        print(f"Confidence: {probabilities[predicted_label_idx]:.2%}")

        # Visualize probability distribution
        plt.figure(figsize=(10, 4))
        plt.bar(range(len(probabilities)), probabilities.cpu().numpy())
        plt.title('Prediction Probabilities')
        plt.xlabel('Class')
        plt.ylabel('Probability')
        plt.xticks(range(len(label_names)), label_names, rotation=45)
        plt.tight_layout()
        plt.show()

def train_model(train_loader, test_loader, device='cuda:0'):
    """
    Train the SNN model.

    Args:
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        device: Device to run the model on (default: 'cuda:0')

    Returns:
        Trained model
    """
    # Create model
    net = Net().to(device)
    
    # Create optimizer and scheduler
    optimizer = Adam(net.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, GAMMA, last_epoch=-1)
    
    # Create loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Create tensorboard writer
    writer = SummaryWriter('./logs/')
    
    # Training loop
    for e in range(NUM_EPOCHS):
        net.train()
        print(f'Epoch {net.epochs}')
        
        time_start = time.time()
        
        # Training phase
        for audios, labels in tqdm(train_loader):
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            out_spikes_counter_frequency = net(audios)
            
            loss = criterion(out_spikes_counter_frequency, labels)
            loss.backward()
            
            optimizer.step()
            
            reset_net(net)
            
            # Rate-based output decoding
            correct_rate = (out_spikes_counter_frequency.argmax(
                dim=1) == labels).float().mean().item()
            
            net.train_times += 1
        
        # Update learning rate
        if e >= WARMUP_EPOCHS:
            lr_scheduler.step()
        
        net.eval()
        
        writer.add_scalar('Train Loss', loss.item(), global_step=net.epochs)
        
        # Testing phase
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0
            pred = []
            label = []
            for audios, labels in tqdm(test_loader):
                audios = audios.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
                
                out_spikes_counter = net(audios)
                
                preds = out_spikes_counter.argmax(dim=1)
                
                correct_sum += (preds == labels).float().sum().item()
                
                pred.append(preds)
                label.append(labels)
                
                test_sum += labels.numel()
                reset_net(net)
            
            pred = torch.cat(pred).cpu().numpy()
            label = torch.cat(label).cpu().numpy()
            
            # Confusion matrix
            cmatrix = confusion_matrix(label, pred)
            
            print("Confusion Matrix:")
            print(cmatrix)
            
            test_accuracy = correct_sum / test_sum
            writer.add_scalar('Test Acc.', test_accuracy, global_step=net.epochs)
        
        net.epochs += 1
        time_end = time.time()
        print(f'Test Acc: {test_accuracy} Loss: {loss} Elapse: {time_end - time_start:.2f}s')
    
    return net 