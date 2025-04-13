# Spiking Neural Network for Speech Recognition

This project implements a Spiking Neural Network (SNN) for speech command recognition using the Speech Commands dataset. The implementation is based on the paper "Technical report: supervised training of convolutional spiking neural networks with PyTorch".

## Project Structure

```
.
├── src/
│   ├── config.py      # Configuration parameters
│   ├── data.py        # Data loading and preprocessing
│   ├── model.py       # SNN model definition
│   ├── train.py       # Training utilities
│   └── utils.py       # Utility functions
├── train_snn.ipynb    # Training notebook
├── requirements.txt   # Project dependencies
└── README.md         # This file
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SNNMusicDetection
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Download the Speech Commands dataset and place it in a `data` directory:
```bash
mkdir data
# Download the dataset to the data directory
```

2. Open and run the training notebook:
```bash
jupyter notebook train_snn.ipynb
```

The notebook will:
- Load and preprocess the data
- Create and train the SNN model
- Evaluate the model and generate a confusion matrix
- Save the best model to `best_model.pth`

## Model Architecture

The model uses a convolutional architecture with Leaky Integrate-and-Fire (LIF) neurons:
- Input: Mel spectrogram of speech commands
- Three convolutional layers with LIF neurons
- Fully connected layer for classification
- Output: 12 classes (speech commands)

## Results

The model achieves comparable performance to traditional artificial neural networks on the Speech Commands dataset. Training progress can be monitored using TensorBoard:

```bash
tensorboard --logdir logs
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.