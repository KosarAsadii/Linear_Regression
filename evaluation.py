import torch

def mae(prediction, out_train):
    return torch.mean(torch.abs(prediction - out_train))

def mse(prediction, out_train):
    return torch.mean((prediction - out_train)**2)

def rmse(prediction, out_train):
    return torch.sqrt(torch.mean((prediction - out_train)**2))

