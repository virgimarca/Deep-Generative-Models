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

class NADE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim

        self.W = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        self.c = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.V = nn.Parameter(torch.zeros(input_dim, hidden_dim, 10))
        self.b = nn.Parameter(torch.zeros(input_dim, 10))

        # He initialization
        nn.init.kaiming_normal_(self.W)
        nn.init.kaiming_normal_(self.V)

    def _forward(self, x):
        original_shape = x.shape
        if len(x.shape) > 2:
            x = x.view(original_shape[0], -1)
        flatten_shape = x.shape

        p_hat, x_hat = [], []
        batch_size = 1 if x is None else x.shape[0]

        a = self.c.expand(-1, batch_size).T
        for i in range(self._input_dim):
            h = torch.sigmoid(a)  # hxb
            p_i = F.softmax(h @ self.V[i: i + 1, :, :].squeeze() + self.b[i: i + 1, :],
                            dim=1)  # bxh @ hx10 + bx10 -> bx10

            p_hat.append(p_i)
            x_i = x[:, i: i + 1]
            if torch.any(x_i < 0):
                x_i = torch.multinomial(p_i, 1)

            x_hat.append(x_i)

            a = a + x_i.float() @ self.W[:, i: i + 1].T

        p_hat, x_hat = torch.stack(p_hat, dim=2), torch.cat(x_hat, dim=1).view(flatten_shape)
        return p_hat, x_hat

    def forward(self, x):
        return self._forward(x)[0]

    def sample(self, n_samples=16, conditioned_on=None):
        with torch.no_grad():
            if not conditioned_on:
                conditioned_on = torch.ones(n_samples, 1, 28, 28) * -1
            return self._forward(conditioned_on)[1]

    def loss(self, x, preds):
        batch_size = x.shape[0]
        x = torch.transpose(F.one_hot(x.view(batch_size, -1), -1), 1, 2).float()
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
