import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    n, c, h, w = x.shape

    for k in range(c):
        bt = x[:, k, :, :]

        mean = torch.mean(bt)
        var = torch.var(bt, correction=0)

        x[:, k, :, :] = gamma[k] * (bt - mean) / torch.sqrt(var + eps) + beta[k]

    return x