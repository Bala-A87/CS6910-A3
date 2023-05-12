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
        self.is_lstm = cell_type == nn.LSTM
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
        self.is_lstm = cell_type == nn.LSTM
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

class AttnDecoder(nn.Module):
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
        self.is_lstm = cell_type == nn.LSTM
        self.embedding = nn.Embedding(num_embeddings=output_size, embedding_dim=embedding_size)
        self.attn = nn.Linear(hidden_size+embedding_size, 100)
        self.attn_combine = nn.Linear(hidden_size+embedding_size, self.hidden_size)
        self.decoder = cell_type(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
        self.out = nn.Linear(in_features=hidden_size, out_features=output_size)
    
    def forward(self, x, hidden, output_enc):
        embedded = self.embedding(x).reshape(1, 1, -1)

        if self.is_lstm:
            attn_wts = F.softmax(self.attn(torch.cat([embedded[0], hidden[0][0]], 1)), dim=1)
        else:
            attn_wts = F.softmax(self.attn(torch.cat([embedded[0], hidden[0]], 1)), dim=1)
        attn_applied = torch.bmm(attn_wts.unsqueeze(0), output_enc.unsqueeze(0))

        output = torch.cat([embedded[0], attn_applied[0]], 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.decoder(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_wts
    
    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)
