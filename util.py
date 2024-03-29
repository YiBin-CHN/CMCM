import torch
import numpy as np
import torch.nn as nn
import copy
import math
from torch.nn import functional as F
from torch.autograd import Variable, Function


def Linear(inputdim, outputdim, bias=True):
    linear = nn.Linear(inputdim, outputdim, bias)
    return linear


def clone(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(nn.Module):
    def __init__(self, head_num, d_model, dropout=0.0):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % head_num == 0
        self.d_k = d_model // head_num
        self.head = head_num
        self.linears = clone(Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def attention(self, q, k, v, mask=None):
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        # print(scores.size(), mask.size())
        # when searching, target mask is not needed
        if mask is not None:
            # b 1 t -> b 1 1 t -> b head t t
            # print(scores.size())
            # print(mask)
            mask = mask.unsqueeze(1).expand_as(scores)
            # print(mask.size())
            scores.masked_fill_(mask == 0, -1e9)
            # print(scores)
        p_att = F.softmax(scores, -1)
        # print(p_att)
        # exit()
        if self.dropout:
            p_att = self.dropout(p_att)
        return torch.matmul(p_att, v)

    def forward(self, query, key, value, mask=None):
        # q k v : B T H
        nbatches = query.size(0)
        # b head t dim
        query, key, value = [l(x).view(nbatches, -1, self.head, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]
        x = self.attention(query, key, value, mask)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.head * self.d_k)
        x = self.linears[-1](x)
        # returen b t dim
        return x

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = Linear(d_model, d_ff)
        self.w_2 = Linear(d_ff, d_model)
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.w_1(x), inplace=True)
        if self.dropout:
            h = self.dropout(h)
        return self.w_2(h)


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        if dropout > 0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
    def forward(self, x, sublayer):
        if self.dropout:
            return self.norm(x + self.dropout(sublayer(x)))
        else:
            return self.norm(x + sublayer(x))

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

def positional_encodings_like(x, t=None):   # hope to be differentiable
    if t is None:
        positions = torch.arange(0, x.size(-2)) # .expand(*x.size()[:2])
        if x.is_cuda:
            positions = positions.cuda(x.get_device())
        positions = Variable(positions.float())
    else:
        positions = t

    # channels
    #channels = torch.arange(0, x.size(-1), 2) / x.size(-1) # 0 2 4 6 ... (256)
    channels = torch.true_divide(torch.arange(0, x.size(-1), 2), x.size(-1))  # 0 2 4 6 ... (256)
    if x.is_cuda:
        channels = channels.cuda(x.get_device())
    #channels = 1 / (10000 ** Variable(channels))
    channels = torch.true_divide(1, 10000 ** Variable(channels))
    channels = channels.float()

    # get the positional encoding: batch x target_len
    encodings = positions.unsqueeze(-1) @ channels.unsqueeze(0)  # batch x target_len x 256
    encodings = torch.cat([torch.sin(encodings).unsqueeze(-1), torch.cos(encodings).unsqueeze(-1)], -1)
    encodings = encodings.contiguous().view(*encodings.size()[:-2], -1)  # batch x target_len x 512

    if encodings.ndimension() == 2:
        encodings = encodings.unsqueeze(0).expand_as(x)

    return encodings


class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, dropout, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(dropout)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))

