"""
Data loading and preprocessing for the SNN speech recognition model.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import Spectrogram
from spikingjelly.datasets.speechcommands import SPEECHCOMMANDS
from scipy.signal import savgol_filter

from .config import N_MELS, F_MAX, F_MIN, DELTA_ORDER, SIZE, LABEL_DICT
from .utils import create_mel_filters

class SpeechCommandDataset(Dataset):
    """Dataset class for speech commands.
    
    This class handles the loading and preprocessing of speech command data
    for training and evaluation.
    """
    
    def __init__(self, root: str, subset: str = 'training'):
        """Initialize the dataset.
        
        Args:
            root: Root directory for the dataset
            subset: Subset of the dataset to use ('training', 'validation', or 'testing')
        """
        self.dataset = SPEECHCOMMANDS(root=root, subset=subset)
        self.spec = Spectrogram(
            sample_rate=16000,
            n_fft=400,
            hop_length=160,
            win_length=400,
            window_fn=torch.hann_window,
            power=2.0,
            normalized=True
        )
        
        # Create mel filterbank
        self.mel_filters = create_mel_filters(
            n_mels=N_MELS,
            n_fft=400,
            f_min=F_MIN,
            f_max=F_MAX,
            sample_rate=16000
        )
        
    def __len__(self) -> int:
        """Get the length of the dataset.
        
        Returns:
            Number of samples in the dataset
        """
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> tuple:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (spectrogram, label)
        """
        # Get audio data and label
        waveform, sample_rate, label, speaker_id, utterance_number = self.dataset[idx]
        
        # Convert label to index
        label = LABEL_DICT[label]
        
        # Compute spectrogram
        spec = self.spec(waveform)
        
        # Apply mel filterbank
        mel_spec = torch.matmul(self.mel_filters, spec)
        
        # Apply log scale
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-10))
        
        # Apply Savitzky-Golay filter for smoothing
        mel_spec = torch.from_numpy(
            savgol_filter(mel_spec.numpy(), window_length=5, polyorder=2, axis=-1)
        )
        
        # Add channel dimension
        mel_spec = mel_spec.unsqueeze(0)
        
        return mel_spec, label

def get_dataloaders(
    root: str,
    batch_size: int = 32,
    num_workers: int = 4
) -> tuple:
    """Create data loaders for training and validation.
    
    Args:
        root: Root directory for the dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = SpeechCommandDataset(root=root, subset='training')
    val_dataset = SpeechCommandDataset(root=root, subset='validation')
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader 