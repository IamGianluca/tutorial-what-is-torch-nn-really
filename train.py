import math
import pickle
import gzip

import numpy as np
import matplotlib
import torch
from matplotlib import pyplot
from constants import PATH, FILENAME


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
        f, encoding="latin-1")


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
n, c = x_train.shape

weights = torch.randn(784, 10) / math.sqrt(784)
weights.requires_grad_()  # we don't want the previos step included in the grad
bias = torch.zeros(10, requires_grad=True)

def log_softmax(x):
    """Activation function."""
    return x - x.exp().sum(-1).log().unsqueeze(-1)

def model(xb):
    """Linear model."""
    return log_softmax(xb @ weights + bias)

bs = 64

xb = x_train[0: bs]
preds = model(xb)
print(preds[1], preds.shape)

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0: bs]
print(loss_func(preds, yb))
