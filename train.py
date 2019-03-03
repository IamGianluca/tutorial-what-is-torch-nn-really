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

def nll(input, target):
    return -input[range(target.shape[0]), target].mean()

loss_func = nll

yb = y_train[0: bs]

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i: end_i]
        yb = y_train[start_i: end_i]
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        with torch.no_grad():  # do not record grad updates
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()

print("loss: ", loss_func(model(xb), yb))
print("accuracy: ", accuracy(model(xb), yb))
