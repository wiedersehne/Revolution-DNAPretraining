import math
import torch
import torch.nn as nn
from linformer import Linformer
from performer_pytorch import Performer
from nystrom_attention import Nystromformer
from torch.nn import TransformerEncoder, TransformerEncoderLayer


#  Refer to https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = x + self.pe[:x.size(0)]  # [seq_len, batch_size, dim]
        x = x.permute(1, 0, 2)
        return self.dropout(x)


class former(nn.Module):
    def __init__(self, model, use_pos, input_size, dim, depth, heads, seq_len=1000):
        super(former, self).__init__()

        self.model = model
        self.use_pos = use_pos
        if use_pos:
            self.linear = nn.Linear(input_size, dim)
            self.pos_enc = PositionalEncoding(dim, seq_len)

        if model == 'transformer':
            encoder_layers = TransformerEncoderLayer(dim, heads, dim)
            self.former = TransformerEncoder(encoder_layers, depth)
        elif model == 'performer':
            self.former = Performer(dim=dim, depth=depth, heads=heads, dim_head=dim)
        elif model == 'linformer':
            self.former = Linformer(dim=dim, seq_len=seq_len, depth=depth, heads=heads, k=dim)
        elif model == 'nystromformer':
            self.former = Nystromformer(dim=dim, depth=depth, heads=heads)

    def forward(self, x):
        if self.use_pos:
            x = self.linear(x)
            x = self.pos_enc(x)  # out: num, length, dim

        if self.model == 'transformer':
            x = x.permute(1, 0, 2)
            x = self.former(x)
            x = x.permute(1, 0, 2)
        else:
            x = self.former(x)

        return x


class Model4Pretrain(nn.Module):
    def __init__(self, model, depth1, depth2, heads, input_size, hdim):
        super().__init__()

        self.encoder = former(model, True, input_size, hdim, depth=depth1, heads=heads)
        self.decoder = former(model, False, hdim, hdim, depth=depth2, heads=heads)
        self.linear = nn.Linear(hdim, input_size)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        logits_lm = self.linear(h)
        return self.sig(logits_lm)


class FORMER(nn.Module):
    def __init__(self, model, depth1, depth2, heads, input_size, hdim, n_targets, seq_len, center=200):
        super().__init__()
        self.seq_len = seq_len
        self.center = center

        self.Net1 = former(model, True, input_size, hdim, depth=depth1, heads=heads)
        self.Net2 = former(model, False, hdim, hdim, depth=depth2, heads=heads)
        self.classifier = nn.Linear(hdim, n_targets)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.Net1(x)
        y = self.Net2(h)
        new_start = int((self.seq_len - self.center) / 2)
        y_new = y[:, new_start:new_start+self.center, :]
        y_class = self.classifier(torch.mean(y_new, dim=1))
        return self.sig(y_class)
