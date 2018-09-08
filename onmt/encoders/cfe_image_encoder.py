""" Image Encoder """
import math

import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.nn.utils import weight_norm

from onmt.encoders.cfe_encoder import reset_parameters, InitialWeights


class MaskedConv2d(nn.Conv2d):
    def __init__(self, right, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + 1:] = 0
        self.mask[:, :, kH // 2 + right:] = 0

        if not right:
            self.mask = 1 - self.mask

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class CausalConv2d(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout, right):
        super(CausalConv2d, self).__init__()

        num_layers = max(1, int(math.ceil(math.log2(receptive_field / (kernel_size - 1) + 1) - 1)))
        layers = [
            weight_norm(MaskedConv2d(right, input_size, feats_size, kernel_size=kernel_size, padding=(kernel_size - 1)//2))
        ]

        for i in range(1, num_layers):
            layers += [
                nn.LeakyReLU(),
                nn.Dropout(dropout),
                weight_norm(MaskedConv2d(right, feats_size, feats_size, kernel_size=kernel_size,
                                      padding=(kernel_size - 1) * 2**(i-1), dilation=2**i))
            ]

        self.net = nn.Sequential(*layers)
        reset_parameters(self, 'conv2d')

    def forward(self, input):
        return self.net(input).squeeze(dim=2)


class CausalFeatureExtractor(nn.Module):
    def __init__(self, input_size, feats_size, kernel_size, receptive_field, dropout):
        super(CausalFeatureExtractor, self).__init__()
        self.input_size = input_size
        self.dropout = dropout
        assert feats_size % 2 == 0, "the number of features has to be an even number"

        self.right = CausalConv2d(input_size, feats_size // 2, kernel_size, receptive_field, dropout, right=True)
        self.left = CausalConv2d(input_size, feats_size // 2, kernel_size, receptive_field, dropout, right=False)

    def forward(self, input):
        out = torch.cat((self.left(input), self.right(input)), dim=1)
        return out


class CFEImageEncoder(nn.Module):
    def __init__(self, receptive_field, hidden_size, kernel_width, rnn_layers, dropout):
        super(CFEImageEncoder, self).__init__()
        self.hidden_size = hidden_size

        self.layer1 = nn.Conv2d(3, 64, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer2 = nn.Conv2d(64, 128, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer3 = nn.Conv2d(128, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer4 = nn.Conv2d(256, 256, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer5 = nn.Conv2d(256, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))
        self.layer6 = nn.Conv2d(512, 512, kernel_size=(3, 3),
                                padding=(1, 1), stride=(1, 1))

        self.batch_norm1 = nn.BatchNorm2d(256)
        self.batch_norm2 = nn.BatchNorm2d(512)
        self.batch_norm3 = nn.BatchNorm2d(512)

        src_size = 512
        self.cfe = CausalFeatureExtractor(src_size, hidden_size, kernel_width, receptive_field, dropout)

        self.linear = nn.Linear(src_size, hidden_size)
        self.init_decoder = InitialWeights(hidden_size, 256, rnn_layers)  # TODO not hardcode the parameters

    def forward(self, src, lengths=None):
        batch_size = src.size(0)
        src = F.relu(self.layer1(src[:, :, :, :] - 0.5), True)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))
        src = F.relu(self.layer2(src), True)
        src = F.max_pool2d(src, kernel_size=(2, 2), stride=(2, 2))
        src = F.relu(self.batch_norm1(self.layer3(src)), True)
        src = F.relu(self.layer4(src), True)
        src = F.max_pool2d(src, kernel_size=(1, 2), stride=(1, 2))
        src = F.relu(self.batch_norm2(self.layer5(src)), True)
        src = F.max_pool2d(src, kernel_size=(2, 1), stride=(2, 1))
        src = F.relu(self.batch_norm3(self.layer6(src)), True)
        # (batch_size, 512, H, W)

        hidden = self.init_decoder(src)
        output = self.cfe(src)
        output = output.view(batch_size, self.hidden_size, -1)
        output = output.transpose(1,2).transpose(0,1).contiguous()

        return hidden, output
