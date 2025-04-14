"""
Configuration parameters for the SNN speech recognition model.
"""

# Dataset parameters
LABEL_DICT = {
    'yes': 0, 'stop': 1, 'no': 2, 'right': 3, 'up': 4, 'left': 5, 
    'on': 6, 'down': 7, 'off': 8, 'go': 9, 'bed': 10, 'three': 10, 
    'one': 10, 'four': 10, 'two': 10, 'five': 10, 'cat': 10, 'dog': 10, 
    'eight': 10, 'bird': 10, 'happy': 10, 'sheila': 10, 'zero': 10, 
    'wow': 10, 'marvin': 10, 'house': 10, 'six': 10, 'seven': 10, 
    'tree': 10, 'nine': 10, '_silence_': 11
}
LABEL_CNT = len(set(LABEL_DICT.values()))

# Audio processing parameters
SAMPLE_RATE = 16000
N_FFT = int(30e-3 * SAMPLE_RATE)  # 30ms window
HOP_LENGTH = int(10e-3 * SAMPLE_RATE)  # 10ms hop
N_MELS = 40
F_MAX = 4000
F_MIN = 20
DELTA_ORDER = 0
SIZE = 16000

# Model parameters
TAU = 10.0 / 7
V_THRESHOLD = 1.0
V_RESET = 0.0
ALPHA = 10.0

# Training parameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-2
NUM_EPOCHS = 1
WARMUP_EPOCHS = 1
GAMMA = 0.85
NUM_WORKERS = 16

# Backend configuration
try:
    import cupy
    BACKEND = 'cupy'
except ModuleNotFoundError:
    BACKEND = 'torch'
    print('Cupy is not installed. Using torch backend for neurons.') 