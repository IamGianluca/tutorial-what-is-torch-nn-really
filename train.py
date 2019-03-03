import math
import pickle
import gzip

import numpy as np
import matplotlib
import torch
import torch.nn.functional as F
from matplotlib import pyplot
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from constants import PATH, FILENAME


with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(
        f, encoding="latin-1")


x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)
bs = 64
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# NOTE: we’ll use a batch size for the validation set that is twice as large
# as that for the training set. This is because the validation set does not
# need backpropagation and thus takes less memory
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)

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

model, opt = get_model()
xb, yb = x_train[0: bs], y_train[0: bs]
print("loss: ", loss_func(model(xb), yb))

def fit():
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            pred = model(xb)
            loss = loss_func(pred, yb)

            loss.backward()
            opt.step()
            opt.zero_grad()

        model.eval()
        with torch.no_grad():
            valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

        print(epoch, valid_loss / len(valid_dl))

fit()

xb, yb = x_train[0: bs], y_train[0: bs]
print("loss: ", loss_func(model(xb), yb))
# print("accuracy: ", accuracy(model(xb), yb))
