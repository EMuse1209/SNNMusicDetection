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
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    dct_type: Optional[str] = 'slaney') -> torch.Tensor:
    """Create mel filterbank matrix.
    
    Args:
        n_freqs: Number of frequency bins
        f_min: Minimum frequency
        f_max: Maximum frequency
        n_mels: Number of mel bands
        sample_rate: Sample rate
        dct_type: Type of DCT ('htk' or 'slaney')
        
    Returns:
        Mel filterbank matrix
    """
    if dct_type != "htk" and dct_type != "slaney":
        raise ValueError("DCT type must be either 'htk' or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f)
    m_min = hz_to_mel(f_min, dct_type)
    m_max = hz_to_mel(f_max, dct_type)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel)
    f_pts = mel_to_hz(m_pts, dct_type)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    # (n_freqs, n_mels + 2)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if dct_type == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb 