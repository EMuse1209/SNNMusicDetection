"""
Training utilities for the SNN speech recognition model.
"""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

from .config import LEARNING_RATE, DEVICE, LABEL_DICT
from .model import SNN
from .data import SpeechCommandDataset

def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = DEVICE
) -> float:
    """Train the model for one epoch.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to use for training
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = DEVICE
) -> tuple:
    """Evaluate the model on the validation set.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to use for evaluation
        
    Returns:
        Tuple of (average validation loss, accuracy)
    """
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    return total_loss / len(val_loader), correct / total

def plot_confusion_matrix(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    device: str = DEVICE
) -> None:
    """Plot confusion matrix for the validation set.
    
    Args:
        model: The model to evaluate
        val_loader: DataLoader for validation data
        device: Device to use for evaluation
    """
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int,
    learning_rate: float = LEARNING_RATE,
    device: str = DEVICE,
    log_dir: str = 'logs'
) -> None:
    """Train the model.
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimization
        device: Device to use for training
        log_dir: Directory for TensorBoard logs
    """
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir)
    
    best_acc = 0
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        writer.add_scalar('Loss/train', train_loss, epoch)
        
        # Evaluate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        print(f'Epoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  Val Accuracy: {val_acc:.4f}')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    writer.close()
    
    # Plot confusion matrix
    plot_confusion_matrix(model, val_loader, device)

def test_single_sample(
    model: nn.Module,
    audio_file: str,
    device: str = DEVICE,
    model_path: str = 'best_model.pth'
) -> tuple:
    """Test the model on a single audio file.
    
    Args:
        model: The model to use for prediction
        audio_file: Path to the audio file
        device: Device to use for inference
        model_path: Path to the saved model weights
        
    Returns:
        Tuple of (predicted_label, confidence)
    """
    # Load model weights if provided
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    
    # Create a dataset with just this file
    dataset = SpeechCommandDataset(root=os.path.dirname(audio_file), subset='testing')
    
    # Find the index of the audio file in the dataset
    file_idx = None
    for i in range(len(dataset)):
        _, _, _, _, utterance_number = dataset.dataset[i]
        if os.path.basename(audio_file) in str(utterance_number):
            file_idx = i
            break
    
    if file_idx is None:
        raise ValueError(f"Audio file {audio_file} not found in the dataset")
    
    # Get the preprocessed audio
    mel_spec, true_label = dataset[file_idx]
    
    # Add batch dimension
    mel_spec = mel_spec.unsqueeze(0).to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    # Make prediction
    with torch.no_grad():
        output = model(mel_spec)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        predicted_label = output.argmax(dim=1).item()
        confidence = probabilities[0, predicted_label].item()
    
    # Get the label name
    label_names = {v: k for k, v in LABEL_DICT.items()}
    predicted_label_name = label_names[predicted_label]
    true_label_name = label_names[true_label]
    
    print(f"True label: {true_label_name}")
    print(f"Predicted label: {predicted_label_name}")
    print(f"Confidence: {confidence:.4f}")
    
    return predicted_label_name, confidence 