"""
Data loading and preprocessing for the SNN speech recognition model.
"""

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchaudio.transforms import Spectrogram
from spikingjelly.datasets.speechcommands import SPEECHCOMMANDS
from scipy.signal import savgol_filter
import numpy as np

from .config import (
    LABEL_DICT, N_MELS, F_MAX, F_MIN, DELTA_ORDER, SIZE,
    SAMPLE_RATE, N_FFT, HOP_LENGTH
)
from .utils import create_mel_filters

class Pad:
    """Pad the audio to a fixed length."""
    def __init__(self, size):
        self.size = size

    def __call__(self, wav):
        wav_size = wav.shape[-1]
        pad_size = (self.size - wav_size) // 2
        padded_wav = nn.functional.pad(
            wav, (pad_size, self.size-wav_size-pad_size), mode='constant', value=0)
        return padded_wav

class MelScaleDelta(nn.Module):
    """Compute mel spectrogram and its temporal derivatives."""
    def __init__(self, order, n_mels=128, sample_rate=16000, f_min=0., f_max=None, dct_type='slaney'):
        super().__init__()
        self.order = order
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.dct_type = dct_type
        
        # Create mel filterbank
        self.mel_filters = create_mel_filters(
            n_freqs=N_FFT // 2 + 1,
            f_min=f_min,
            f_max=self.f_max,
            n_mels=n_mels,
            sample_rate=sample_rate,
            dct_type=dct_type
        )
        self.register_buffer('filters', self.mel_filters)

    def forward(self, specgram):
        # Apply mel filterbank
        mel_specgram = torch.matmul(self.filters, specgram)
        
        # Apply log scale and normalization
        M = torch.max(torch.abs(mel_specgram))
        if M > 0:
            feat = torch.log1p(mel_specgram/M)
        else:
            feat = mel_specgram

        # Compute temporal derivatives
        feat_list = [feat.numpy().T]
        for k in range(1, self.order + 1):
            feat_list.append(savgol_filter(
                feat.numpy(), 9, deriv=k, axis=-1, mode='interp', polyorder=k).T)

        return torch.as_tensor(np.expand_dims(np.stack(feat_list), axis=0))

class Rescale:
    """Rescale the features by their standard deviation."""
    def __call__(self, input):
        std = torch.std(input, axis=2, keepdims=True, unbiased=False)
        std.masked_fill_(std == 0, 1)
        return input / std

def collate_fn(data):
    """Custom collate function for batch creation."""
    X_batch = torch.cat([d[0] for d in data])
    std = X_batch.std(axis=(0, 2), keepdim=True, unbiased=False)
    X_batch.div_(std)
    y_batch = torch.tensor([d[1] for d in data])
    return X_batch, y_batch

def get_dataloaders(
    root: str,
    batch_size: int = 64,
    num_workers: int = 16
) -> tuple:
    """Create data loaders for training, validation and testing.
    
    Args:
        root: Root directory for the dataset
        batch_size: Batch size for the data loaders
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create transforms
    pad = Pad(SIZE)
    spec = Spectrogram(
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        power=2.0
    )
    melscale = MelScaleDelta(
        order=DELTA_ORDER,
        n_mels=N_MELS,
        sample_rate=SAMPLE_RATE,
        f_min=F_MIN,
        f_max=F_MAX,
        dct_type='slaney'
    )
    rescale = Rescale()
    
    transform = nn.Sequential(pad, spec, melscale, rescale)
    
    # Create datasets
    train_dataset = SPEECHCOMMANDS(
        label_dict=LABEL_DICT,
        root=root,
        silence_cnt=2300,
        url="speech_commands_v0.01",
        split="train",
        transform=transform,
        download=True
    )
    
    test_dataset = SPEECHCOMMANDS(
        label_dict=LABEL_DICT,
        root=root,
        silence_cnt=260,
        url="speech_commands_v0.01",
        split="test",
        transform=transform,
        download=True
    )
    
    # Create samplers
    train_sampler = WeightedRandomSampler(
        train_dataset.weights,
        len(train_dataset.weights)
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader 