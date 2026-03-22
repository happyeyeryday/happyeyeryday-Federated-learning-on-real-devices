import copy
import torch
import numpy as np


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


def accuracy(y_hat, y):
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


class Accumulator:
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def uniform_distribution_loss(a, b=None):
    if b is None:
        b = np.zeros_like(a)
    sum_vec = a + b
    if np.sum(sum_vec) == 0:
        return 0
    sum_vec = sum_vec / np.sum(sum_vec)
    uniform_vec = np.array([1 / len(sum_vec) for _ in range(len(sum_vec))])
    return np.linalg.norm(sum_vec - uniform_vec)
    
def kl_divergence(a, b):
    return abs(sum(a[i] * np.log(a[i]/b[i]+1/1e5) for i in range(len(a))))
    
def KL_loss(d, args):
    d_0 = np.full(args.num_classes, 1)
    d = d / np.sum(d)
    d_0 = d_0 / np.sum(d_0)
    return kl_divergence(d,d_0)
