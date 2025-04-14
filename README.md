# SNN Speech Recognition

This project implements a Spiking Neural Network (SNN) for speech recognition using the Speech Commands dataset. The implementation is based on the paper "Technical report: supervised training of convolutional spiking neural networks with PyTorch" (https://arxiv.org/pdf/1911.10124.pdf).

## Project Structure

The project is organized into the following modules:

- `src/config.py`: Configuration parameters for the model
- `src/utils.py`: Utility functions for mel filterbank creation
- `src/data.py`: Data loading and preprocessing
- `src/model.py`: Neural network model definition
- `src/train.py`: Training and evaluation functions
- `train_snn.ipynb`: Jupyter notebook for training and testing

## Requirements

- Python 3.8+
- PyTorch
- spikingjelly (from GitHub)
- torchaudio
- scipy
- matplotlib
- sklearn
- tqdm

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/snn-speech-recognition.git
cd snn-speech-recognition
```

2. Install the required packages:
```bash
pip install -r requirements.txt
pip install --upgrade "https://github.com/fangwei123456/spikingjelly/tarball/master"
```

## Usage

1. Open the `train_snn.ipynb` notebook in Jupyter:
```bash
jupyter notebook train_snn.ipynb
```

2. Run the cells in the notebook to train the model and test it on a single sample.

## Model Architecture

The model uses a convolutional architecture with Leaky Integrate-and-Fire (LIF) neurons:

1. Input: Mel spectrogram with temporal derivatives
2. Three convolutional layers with LIF neurons
3. Fully connected layer for classification

## Results

The model achieves comparable performance to traditional artificial neural networks on the Speech Commands dataset.

## License

This project is licensed under the MIT License - see the LICENSE file for details.