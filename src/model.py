    """
    Neural network model for the SNN speech recognition.
    """

    import torch
    from torch import nn
    from spikingjelly.activation_based import neuron, surrogate
    from spikingjelly.activation_based.functional import reset_net

    from .config import TAU, V_THRESHOLD, V_RESET, ALPHA, BACKEND, LABEL_CNT

    class LIFWrapper(nn.Module):
        """Wrapper for LIF neurons to handle temporal sequences."""
        def __init__(self, module, flatten=False):
            super().__init__()
            self.module = module
            self.flatten = flatten

        def forward(self, x_seq: torch.Tensor):
            """
            Args:
                x_seq: shape=[batch size, channel, T, n_mel]
                
            Returns:
                y_seq: shape=[batch size, channel, T, n_mel] or [batch size, T, channel * n_mel]
            """
            # Input: [batch size, channel, T, n_mel]
            y_seq = self.module(x_seq.transpose(0, 2))  # [T, channel, batch size, n_mel]
            if self.flatten:
                y_seq = y_seq.permute(2, 0, 1, 3)  # [batch size, T, channel, n_mel]
                shape = y_seq.shape[:2]
                return y_seq.reshape(shape + (-1,))  # [batch size, T, channel * n_mel]
            else:
                return y_seq.transpose(0, 2)  # [batch size, channel, T, n_mel]

    class Net(nn.Module):
        """Spiking Neural Network for speech recognition."""
        def __init__(self):
            super().__init__()

            self.train_times = 0
            self.epochs = 0
            self.max_test_acccuracy = 0

            # batch size * delta_order+1 * T * n_mel
            self.conv = nn.Sequential(
                # 101 * 40
                nn.Conv2d(in_channels=1, out_channels=64,
                          kernel_size=(4, 3), stride=1, padding=(2, 1), bias=False),
                LIFWrapper(neuron.LIFNode(
                    tau=TAU,
                    v_threshold=V_THRESHOLD,
                    v_reset=V_RESET,
                    surrogate_function=surrogate.Sigmoid(alpha=ALPHA),
                    backend=BACKEND,
                    step_mode='m'
                )),

                # 102 * 40
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=(4, 3), stride=1, padding=(6, 3), dilation=(4, 3), bias=False),
                LIFWrapper(neuron.LIFNode(
                    tau=TAU,
                    v_threshold=V_THRESHOLD,
                    v_reset=V_RESET,
                    surrogate_function=surrogate.Sigmoid(alpha=ALPHA),
                    backend=BACKEND,
                    step_mode='m'
                )),

                # 102 * 40
                nn.Conv2d(in_channels=64, out_channels=64,
                          kernel_size=(4, 3), stride=1, padding=(24, 9), dilation=(16, 9), bias=False),
                LIFWrapper(neuron.LIFNode(
                    tau=TAU,
                    v_threshold=V_THRESHOLD,
                    v_reset=V_RESET,
                    surrogate_function=surrogate.Sigmoid(alpha=ALPHA),
                    backend=BACKEND,
                    step_mode='m'
                ), flatten=True),
            )
            # [batch size, T, channel * n_mel]
            self.fc = nn.Linear(64 * 40, LABEL_CNT)

        def forward(self, x):
            x = self.fc(self.conv(x))  # [batch size, T, #Class]
            return x.mean(dim=1)  # [batch size, #Class] 