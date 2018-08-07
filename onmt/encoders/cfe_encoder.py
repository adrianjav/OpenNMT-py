import math
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from onmt.encoders.encoder import EncoderBase


def reset_parameters(module, gain):
    for name, param in module.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0.0)
        elif 'weight' in name:
            nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain(gain))


class InitialWeights(nn.Module):
    def __init__(self, hidden_size, num_linear_hidden, lstm_layers):
        super(InitialWeights, self).__init__()
        self.lstm_layers = lstm_layers

        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.net = nn.Sequential(
            nn.Linear(1, num_linear_hidden), nn.LeakyReLU(),
            nn.Linear(num_linear_hidden, num_linear_hidden), nn.LeakyReLU(),
            nn.Linear(num_linear_hidden, hidden_size * 2 * lstm_layers)
        )

        reset_parameters(self, 'leaky_relu')

    def forward(self, feats):
        feats = self.pool(feats).squeeze(1)
        out = self.net(feats).chunk(2, dim=1)
        out = [x.chunk(self.lstm_layers, dim=1) for x in out]
        return out


class Chomp(nn.Module):
    def __init__(self, chomp_size, right=True):
        super(Chomp, self).__init__()
        self.chomp_size = chomp_size
        self.right = right

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.right else x[..., self.chomp_size:]


class CausalConv2d(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout, right):
        super(CausalConv2d, self).__init__()
        self.input_size = input_size
        self.feats_size = feats_size
        self.receptive_field = receptive_field
        self.dropout = dropout
        self.right = right

        num_layers = max(1, int(math.ceil(math.log2(receptive_field / (kernel_size - 1) + 1) - 1)))
        layers = [
            weight_norm(nn.Conv2d(1, feats_size, kernel_size=(input_size, kernel_size),
                                  padding=(0, (kernel_size - 1)))),
            Chomp(kernel_size - 1, right)
        ]

        for i in range(1, num_layers):
            layers += [
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                weight_norm(nn.Conv2d(feats_size, feats_size, kernel_size=(1, kernel_size),
                                      padding=(0, (kernel_size - 1) * 2**i), dilation=(1, 2**i))),
                Chomp((kernel_size - 1) * 2**i, right)
            ]

        self.net = nn.Sequential(*layers)
        reset_parameters(self, 'conv2d')

    def forward(self, input):
        return self.net(input.unsqueeze(dim=1)).squeeze(dim=2)


class CausalFeatureExtractor(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout):
        super(CausalFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        assert feats_size % 2 == 0, "the number of features has to be an even number"

        self.right = CausalConv2d(input_size, feats_size // 2, kernel_size, receptive_field, dropout, right=True)
        self.left = CausalConv2d(input_size, feats_size // 2, kernel_size, receptive_field, dropout, right=False)

    def forward(self, input):
        return torch.cat((self.left(input), self.right(input)), dim=1)


class CFEEncoder(EncoderBase):
    def __init__(self, receptive_field, hidden_size, kernel_width, rnn_layers, dropout, embeddings):
        super(CFEEncoder, self).__init__()
        input_size = embeddings.embedding_size

        self.embeddings = embeddings
        self.linear = nn.Linear(input_size, hidden_size)

        self.cfe = CausalFeatureExtractor(input_size, hidden_size, kernel_width, receptive_field, dropout)
        self.init_decoder = InitialWeights(hidden_size, 256, rnn_layers)  # TODO not hardcoded parameter

    def forward(self, src, lengths=None):
        self._check_args(src, lengths)  # src l x b x v

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        emb = emb.transpose(0, 1).contiguous()
        emb_remap = self.linear(emb).transpose(1,2).unsqueeze(2)
        out = self.cfe(emb_remap)

        return self.init_decoder(emb_remap), \
               out.squeeze(2).transpose(0, 1).transpose(1,2).contiguous()
