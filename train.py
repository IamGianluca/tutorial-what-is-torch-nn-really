import math
import pickle
import gzip

import numpy as np
import matplotlib
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import TensorDataset

from constants import PATH, FILENAME


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
        f, encoding="latin-1")


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
train_ds = TensorDataset(x_train, y_train)

n, c = x_train.shape
bs = 64

loss_func = F.cross_entropy

def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()

class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)

model = MnistLogistic()
lr = 0.5
epochs = 2

def get_model():
    model = MnistLogistic()
    return model, optim.SGD(model.parameters(), lr=lr)

xb, yb = train_ds[0: bs]

model, opt = get_model()
print("loss: ", loss_func(model(xb), yb))

def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):
            xb, yb = train_ds[i*bs: i*bs+bs]
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

fit()

print("loss: ", loss_func(model(xb), yb))
print("accuracy: ", accuracy(model(xb), yb))
