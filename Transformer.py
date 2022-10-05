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

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for _ in range(1)])
        self.input_dropout = nn.Dropout(p=0.3)

        self.embeds = nn.Embedding(10, 100)
        self.output_dense = nn.Linear(100, 10, bias=True)

    def shift_and_pad_(self, X):
        # Shift inputs over by 1 and pad
        shape = X.shape
        X = X.view(shape[0], shape[1] * shape[2], shape[3])
        X = X[:, :-1, :]
        X = F.pad(X, (0, 0, 1, 0))  # Pad second to last dimension
        X = X.view(shape)
        return X

    def forward(self, X):

        X = X.permute([0, 2, 3, 1]).contiguous()
        X = X.view(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])  # Flatten channels into width

        X = self.embeds(X.int()) * (100 ** 0.5)
        X = self.shift_and_pad_(X)
        shape = X.shape
        X = X.view(shape[0], -1, shape[3])

        X = self.input_dropout(X)
        for layer in self.layers:
            X = layer(X)
        X = self.output_dense(X).view(shape[:3] + (-1,))

        X = X.view(X.shape[0], X.shape[1], X.shape[2] //, 1, X.shape[3])
        X = X.permute([0, 3, 1, 2, 4])

        return F.softmax(X, dim=-1)

    def loss(self, x, preds):
        batch_size = x.shape[0]
        x = torch.transpose(F.one_hot(x.view(batch_size, -1), -1), 1, 2).float()
        preds = preds.squeeze().transpose(3, 1).reshape(batch_size, 10, -1)
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

    def sample(self, n_samples=16):
        with torch.no_grad():
            conditioned_on = torch.ones(n_samples, 1, 28, 28)
            return self.forward(conditioned_on)


class DecoderLayer(nn.Module):
    """Implements a single layer of an unconditional ImageTransformer"""

    def __init__(self):
        super().__init__()
        self.attn = Attn()
        self.dropout = nn.Dropout(p=0.3)
        self.layernorm_attn = nn.LayerNorm([100], eps=1e-6, elementwise_affine=True)
        self.layernorm_ffn = nn.LayerNorm([100], eps=1e-6, elementwise_affine=True)
        self.ffn = nn.Sequential(nn.Linear(100, 2048, bias=True),
                                 nn.ReLU(),
                                 nn.Linear(2048, 100, bias=True))

    # Takes care of the "postprocessing" from tensorflow code with the layernorm and dropout
    def forward(self, X):
        y = self.attn(X)
        X = self.layernorm_attn(self.dropout(y) + X)
        y = self.ffn(X)
        X = self.layernorm_ffn(self.dropout(y) + X)
        return X


class Attn(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.kd = 100
        self.vd = 100
        self.q_dense = nn.Linear(100, self.kd, bias=False)
        self.k_dense = nn.Linear(100, self.kd, bias=False)
        self.v_dense = nn.Linear(100, self.vd, bias=False)
        self.output_dense = nn.Linear(self.vd, 100, bias=False)
        assert self.kd % 2 == 0
        assert self.vd % 2 == 0

    def dot_product_attention(self, q, k, v, bias=None):
        logits = torch.einsum("...kd,...qd->...qk", k, q)
        if bias is not None:
            logits += bias
        weights = F.softmax(logits, dim=-1)
        return weights @ v

    def forward(self, X):
        q = self.q_dense(X)
        k = self.k_dense(X)
        v = self.v_dense(X)
        # Split to shape [batch_size, num_heads, len, depth / num_heads]
        q = q.view(q.shape[:-1] + (2, self.kd // 2)).permute([0, 2, 1, 3])
        k = k.view(k.shape[:-1] + (2, self.kd // 2)).permute([0, 2, 1, 3])
        v = v.view(v.shape[:-1] + (2, self.vd // 2)).permute([0, 2, 1, 3])
        q *= (self.kd // 2) ** (-0.5)

        len = X.shape[1]
        blen = 256
        pad = (0, 0, 0, (-len) % 256)  # Append to multiple of block length
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)

        bias = -1e9 * torch.triu(torch.ones(blen, blen), 1).to(X.device)
        first_output = self.dot_product_attention(
            q[:, :, :blen, :], k[:, :, :blen, :], v[:, :, :blen, :], bias=bias)

        if q.shape[2] > blen:
            q = q.view(q.shape[0], q.shape[1], -1, blen, q.shape[3])
            k = k.view(k.shape[0], k.shape[1], -1, blen, k.shape[3])
            v = v.view(v.shape[0], v.shape[1], -1, blen, v.shape[3])
            local_k = torch.cat([k[:, :, :-1], k[:, :, 1:]], 3)  # [batch, nheads, (nblocks - 1), blen * 2, depth]
            local_v = torch.cat([v[:, :, :-1], v[:, :, 1:]], 3)
            tail_q = q[:, :, 1:]
            bias = -1e9 * torch.triu(torch.ones(blen, 2 * blen), blen + 1).to(X.device)
            tail_output = self.dot_product_attention(tail_q, local_k, local_v, bias=bias)
            tail_output = tail_output.view(tail_output.shape[0], tail_output.shape[1], -1, tail_output.shape[4])
            result = torch.cat([first_output, tail_output], 2)
            result = result[:, :, :X.shape[1], :]
        else:
            result = first_output[:, :, :X.shape[1], :]

        result = result.permute([0, 2, 1, 3]).contiguous()
        result = result.view(result.shape[0:2] + (-1,))
        result = self.output_dense(result)
        return result