import torch
import numpy as np
import torch.nn as nn
from util import clone, MultiHeadedAttention, PositionwiseFeedForward, SublayerConnection
from util import positional_encodings_like, ResidualBlock, PositionalEncoding
from torch.autograd import Variable, Function
import torchvision.models as models
import math

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clone(layer, N)
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward"

    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout)

        self.sublayer = clone(SublayerConnection(d_model, dropout), 2)
        self.size = d_model

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clone(layer, N)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return x

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward"

    def __init__(self, d_model, n_heads, d_hidden, dropout=0.1, positional=None):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.positional = positional
        self.src_attn = MultiHeadedAttention(n_heads, d_model, dropout)
        self.pos = PositionalEncoding(d_model, dropout, max_len=5000)
        if positional:
            self.pos_selfattn = ResidualBlock(
                MultiHeadedAttention(n_heads, d_model, dropout),
                d_model, dropout, pos=2)
        self.feed_forward = PositionwiseFeedForward(d_model, d_hidden, dropout)
        self.sublayer = clone(SublayerConnection(d_model, dropout), 4)

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))

        if self.positional:
            pos_encoding = positional_encodings_like(x)
            x = self.sublayer[1](x, lambda x: self.pos_selfattn(pos_encoding, pos_encoding, x, tgt_mask))

        x = self.sublayer[2](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[3](x, self.feed_forward)

class EncoderCNN(nn.Module):
    def __init__(self, target_size):
        super(EncoderCNN, self).__init__()

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        for param in self.resnet.parameters():
            param.requires_grad = False

        self.linear = nn.Linear(resnet.fc.in_features, target_size)
        self.bn = nn.BatchNorm1d(target_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


class Encoderimage(nn.Module):
    def __init__(self, img_feature_size):
        super(Encoderimage, self).__init__()
        self.cnn = EncoderCNN(img_feature_size)

    def forward(self, story_images):
        data_size = story_images.size()
        local_cnn = self.cnn(story_images.view(-1, data_size[2], data_size[3], data_size[4])).view(data_size[0], data_size[1], -1)

        return local_cnn



