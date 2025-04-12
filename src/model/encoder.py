import torch
import torch.nn as nn

class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)

    def forward(self, x):
        # x: (batch_size, seq_len, input_size)
        outputs, (hidden, cell) = self.lstm(x)
        return outputs, hidden, cell
