import torch
import torch.nn as nn

class DecoderLSTM(nn.Module):
    def __init__(self, output_size, hidden_size, num_layers, dropout):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

    def forward(self, input_token, hidden, cell):
        # input_token: (batch_size) => needs to be (batch_size, 1)
        input_token = input_token.unsqueeze(1)
        embedded = self.embedding(input_token)  # (batch_size, 1, hidden_size)

        outputs, (hidden, cell) = self.lstm(embedded, (hidden, cell))  # (batch_size, 1, hidden_size)
        predictions = self.fc_out(outputs.squeeze(1))  # (batch_size, output_size)

        return predictions, hidden, cell
