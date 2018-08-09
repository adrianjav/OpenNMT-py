""" CFE audio encoder """
import math
import torch.nn as nn
import torch.nn.functional as F

from onmt.encoders.cfe_encoder import CausalFeatureExtractor, InitialWeights


class CFEAudioEncoder(nn.Module):
    def __init__(self, receptive_field, hidden_size, kernel_width, rnn_layers, dropout, sample_rate, window_size):
        super(CFEAudioEncoder, self).__init__()
        input_size = int(math.floor((sample_rate * window_size) / 2) + 1)
        input_size = int(math.floor(input_size - 41) / 2 + 1)
        input_size = int(math.floor(input_size - 21) / 2 + 1)
        input_size *= 32

        self.layer1 = nn.Conv2d(1, 32, kernel_size=(41, 11), padding=(0, 10), stride=(2, 2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.layer2 = nn.Conv2d(32, 32, kernel_size=(21, 11), padding=(0, 0), stride=(2, 1))
        self.batch_norm2 = nn.BatchNorm2d(32)

        self.cfe = CausalFeatureExtractor(input_size, hidden_size, kernel_width, receptive_field, dropout)

        self.linear = nn.Linear(input_size, hidden_size)
        self.init_decoder = InitialWeights(hidden_size, 256, rnn_layers)  # TODO not hardcode the parameters

    def forward(self, src, lengths=None):
        hidden = self.init_decoder(src)

        src = self.batch_norm1(self.layer1(src[...]))
        src = F.hardtanh(src, 0, 20, inplace=True)
        src = self.batch_norm2(self.layer2(src))
        src = F.hardtanh(src, 0, 20, inplace=True)

        batch_size = src.size(0)
        length = src.size(3)
        src = src.view(batch_size, -1, length)
        output = self.cfe(src)
        output = output.transpose(1,2).transpose(0,1).contiguous()

        return hidden, output
