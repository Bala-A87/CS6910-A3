import torch
from torch import nn
from torch.functional import F

class Encoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        embedding_size: int,
        hidden_size: int,
        cell_type: nn.Module = nn.RNN,
        num_layers: int = 1,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=input_size, embedding_dim=embedding_size)
        self.encoder = cell_type(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout) 
    
    def forward(self, x, hidden):
        output = self.embedding(x).reshape(1, 1, -1) 
        output, hidden = self.encoder(output, hidden)
        return output, hidden
 
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
    
class Decoder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_size: int,
        output_size: int,
        cell_type: nn.Module = nn.RNN,
        num_layers: int = 1,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size)
        self.decoder = cell_type(input_size=embedding_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout) 
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
        self.softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, x, hidden):
        output = self.embedding(x).reshape(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.decoder(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
