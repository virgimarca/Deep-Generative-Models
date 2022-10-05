import numpy as np
from matplotlib import pyplot as plt

import torch
import torchvision
from torch import optim
from torch import nn
from torch import distributions
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import transforms as T


def _padding(i, o, k, s=1, d=1):
    return ((o - 1) * s + (k - 1) * (d - 1) + k - i) // 2



class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, mask='B', **kargs):
        super(MaskedConv2d, self).__init__(*args, **kargs)
        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)

        _, _, H, W = self.mask.size()

        self.mask[:, :, H // 2, W // 2 + (self.mask_type == 'B'):] = 0
        self.mask[:, :, H // 2 + 1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class MaskedConv1d(nn.Conv1d):
    def __init__(self, *args, mask='B', **kargs):
        super(MaskedConv1d, self).__init__(*args, **kargs)
        self.mask_type = mask
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)

        _, _, W = self.mask.size()

        self.mask[:, :, W // 2 + (self.mask_type == 'B'):] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv1d, self).forward(x)


class RowLSTMCell(nn.Module):
    def __init__(self, hidden_dims, image_size, channel_in, *args, **kargs):
        super(RowLSTMCell, self).__init__(*args, **kargs)

        self._hidden_dims = hidden_dims
        self._image_size = image_size
        self._channel_in = channel_in
        self._num_units = self._hidden_dims * self._image_size
        self._output_size = self._num_units
        self._state_size = self._num_units * 2

        self.conv_i_s = MaskedConv1d(self._hidden_dims, 4 * self._hidden_dims, 3, mask='B',
                                     padding=_padding(image_size, image_size, 3))
        self.conv_s_s = nn.Conv1d(channel_in, 4 * self._hidden_dims, 3, padding=_padding(image_size, image_size, 3))

    def forward(self, inputs, states):
        c_prev, h_prev = states

        h_prev = h_prev.view(-1, self._hidden_dims, self._image_size)
        inputs = inputs.view(-1, self._channel_in, self._image_size)

        s_s = self.conv_s_s(h_prev)
        i_s = self.conv_i_s(inputs)

        s_s = s_s.view(-1, 4 * self._num_units)
        i_s = i_s.view(-1, 4 * self._num_units)

        lstm = s_s + i_s

        lstm = torch.sigmoid(lstm)

        i, g, f, o = torch.split(lstm, (4 * self._num_units) // 4, dim=1)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)

        new_state = (c, h)
        return h, new_state


class RowLSTM(nn.Module):
    def __init__(self, hidden_dims, input_size, channel_in, *args, **kargs):
        super(RowLSTM, self).__init__(*args, **kargs)
        self._hidden_dims = hidden_dims
        self.init_state = (torch.zeros(1, input_size * hidden_dims), torch.zeros(1, input_size * hidden_dims))

        self.lstm_cell = RowLSTMCell(hidden_dims, input_size, channel_in)

    def forward(self, inputs):

        n_batch, channel, n_seq, width = inputs.size()
        hidden_init, cell_init = self.init_state


        states = (hidden_init.repeat(n_batch, 1), cell_init.repeat(n_batch, 1))

        steps = []
        for seq in range(n_seq):
            h, states = self.lstm_cell(inputs[:, :, seq, :], states)
            steps.append(h.unsqueeze(1))

        return torch.cat(steps, dim=1).view(-1, n_seq, width, self._hidden_dims).permute(0, 3, 1, 2)


class PixelRNN(nn.Module):
    def __init__(self, num_layers, hidden_dims, input_size, *args, **kargs):
        super(PixelRNN, self).__init__(*args, **kargs)
        pad_conv1 = _padding(input_size, input_size, 7)
        self.conv1 = MaskedConv2d(1, hidden_dims, (7, 7), mask='A', padding=(pad_conv1, pad_conv1))
        self.lstm_list = nn.ModuleList([RowLSTM(hidden_dims, input_size, hidden_dims) for _ in range(num_layers)])
        self.linear = nn.Linear(hidden_dims, 10)

    def forward(self, inputs):
        x = self.conv1(inputs)
        for lstm in self.lstm_list:
            x = lstm(x)
        x = self.linear(x.transpose(1, 3))
        x = torch.sigmoid(x.transpose(1, 3))
        return x

    def loss(self, x, preds):
        batch_size = x.shape[0]
        x = torch.transpose(F.one_hot(x.view(batch_size, -1), -1), 1, 2).float()
        preds = preds.reshape(batch_size, 10, -1)
        loss = F.cross_entropy(preds, x, reduction='sum')
        return loss / batch_size

    def evaluation(self, test_loader):
        test_loss = []
        for data, _ in test_loader:
            x = torch.Tensor(data)
            y = data.clone()

            with torch.no_grad():
                output = self(x.float())
                test_loss.append(self.loss(y, output))

        return np.mean(test_loss)


    def sample(self, n_samples=16, conditioned_on=None):
        with torch.no_grad():
            conditioned_on = torch.ones(n_samples, 1, 28, 28) * -1
            return self.forward(conditioned_on)