"""
Utility functions for the SNN speech recognition model.
"""

import math
import torch
from typing import Union, Optional

def mel_to_hz(mels: Union[torch.Tensor, float], dct_type: str = 'slaney') -> Union[torch.Tensor, float]:
    """Convert mel scale to Hz.
    
    Args:
        mels: Mel scale values
        dct_type: Type of DCT to use ('htk' or 'slaney')
        
    Returns:
        Frequency values in Hz
    """
    if dct_type == 'htk':
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(mels) and mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * torch.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))

    return freqs

def hz_to_mel(frequencies: Union[torch.Tensor, float], dct_type: str = 'slaney') -> Union[torch.Tensor, float]:
    """Convert Hz to mel scale.
    
    Args:
        frequencies: Frequency values in Hz
        dct_type: Type of DCT to use ('htk' or 'slaney')
        
    Returns:
        Mel scale values
    """
    if dct_type == 'htk':
        if torch.is_tensor(frequencies) and frequencies.ndim:
            return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
        return 2595.0 * math.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(frequencies) and frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + torch.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, check directly
        mels = min_log_mel + math.log(frequencies / min_log_hz) / logstep

    return mels

def create_mel_filters(
    n_mels: int,
    n_fft: int,
    f_min: float,
    f_max: float,
    sample_rate: int,
    dct_type: str = 'slaney'
) -> torch.Tensor:
    """Create mel filterbank matrix.
    
    Args:
        n_mels: Number of mel bands
        n_fft: Number of FFT components
        f_min: Minimum frequency
        f_max: Maximum frequency
        sample_rate: Sample rate of the audio
        dct_type: Type of DCT to use ('htk' or 'slaney')
        
    Returns:
        Mel filterbank matrix
    """
    # Convert frequencies to mel scale
    f_min_mel = hz_to_mel(f_min, dct_type)
    f_max_mel = hz_to_mel(f_max, dct_type)
    mels = torch.linspace(f_min_mel, f_max_mel, n_mels + 1)
    
    # Convert mel scale back to Hz
    freqs = mel_to_hz(mels, dct_type)
    
    # Convert frequencies to FFT bin numbers
    bins = torch.floor((n_fft + 1) * freqs / sample_rate)
    
    # Create filterbank matrix
    fbank = torch.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        for j in range(int(bins[i]), int(bins[i + 1])):
            fbank[i, j] = (j - bins[i]) / (bins[i + 1] - bins[i])
        for j in range(int(bins[i + 1]), int(bins[i + 2])):
            fbank[i, j] = (bins[i + 2] - j) / (bins[i + 2] - bins[i + 1])
            
    return fbank 