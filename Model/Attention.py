import math
import torch
from torch import nn, Tensor
import torch.utils.data


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class SELayer(nn.Module):
    def __init__(self, c, r=4, use_max_pooling=False):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1) if not use_max_pooling else nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, s, h = x.shape
        y = self.squeeze(x).view(bs, s)
        y = self.excitation(y).view(bs, s, 1)
        return x * y.expand_as(x)


# TS-TSSA
class TS_TSSA(nn.Module):
    def __init__(self, d_model: int, embed_dim, num_head, num_layer, output_n=25, input_n=25, dropout=0.5, dff=2048,
                 device='cuda:0'):
        super(TS_TSSA, self).__init__()
        self.num_layers = num_layer

        self.pos_one = PositionalEncoding(embed_dim)
        self.conv = nn.Conv2d(1, d_model, (1, embed_dim), stride=1)

        self.lin_in = nn.Linear(embed_dim, d_model)
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_head)
        self.transEncoder = nn.TransformerEncoder(self.encoder_layer, num_layer)
        self.MHA = nn.MultiheadAttention(d_model, num_head)

        self.norm = nn.LayerNorm(d_model)
        self.channel_MLP = nn.Sequential(
            nn.Conv1d(input_n, dff, 1, stride=1),
            nn.ReLU(),
            nn.Conv1d(dff, input_n, 1, stride=1),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.se = SELayer(input_n, r=4, use_max_pooling=False)
        self.conv_out = nn.Conv1d(input_n, output_n, 1, stride=1)
        self.lin_out = nn.Linear(d_model, embed_dim)
        self.mlp_exit = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.ReLU(),
            nn.Linear(dff, d_model),
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        self.lin_one = nn.Linear(d_model, dff)
        self.lin_two = nn.Linear(dff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pos_one(x)
        x = x.unsqueeze(1)
        x = self.conv(x)

        x = x.squeeze(dim=3).transpose(1, 2)

        for _ in range(self.num_layers):
            y = self.channel_MLP(x)
            y = self.se(y)
            x = x + y

        for _ in range(self.num_layers):
            y, _ = self.MHA(x, x, x)  # self.transEncoder(x)
            x = x + self.norm(y)

            y = self.lin_one(x)
            y = self.relu(y)
            y = x + self.norm(y)
            y = self.lin_two(y)
            y = self.relu(y)

            y = self.se(y)
            x = x + y

        for _ in range(self.num_layers):
            x = self.mlp_exit(x)

        x = self.conv_out(x)
        out = self.lin_out(x)

        return out
