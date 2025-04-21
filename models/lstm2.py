from torch import nn


class TinyLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):
        lstm_out, _ = self.lstm(x_seq)  # lstm_out: [B, T, H]
        out = self.fc(lstm_out)         # out: [B, T, 1]
        return out.squeeze(-1)          # return shape: [B, T]