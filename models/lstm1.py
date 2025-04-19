from torch import nn
from models.mlps import MLP_liststyle

class MyLSTM(nn.Module):
    def __init__(self,
                 InFeatures: int,
                 OutFeatures: int = 1,
                 num_layers=1,
                 HiddenDim : int = 128,
                 FeedForwardDim: int = 256,
                 nonlinearity = 'relu'):
        super().__init__()
        self.lstm = nn.LSTM(input_size=InFeatures,
                            hidden_size=HiddenDim,
                            num_layers=num_layers,
                            batch_first=True)
        self.linear = MLP_liststyle(HiddenDim, OutFeatures, [FeedForwardDim], nonlinearity)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

