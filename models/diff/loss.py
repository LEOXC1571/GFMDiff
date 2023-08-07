
import math
import torch


def gaussian_kl(q_mu, q_sigma, p_mu, p_sigma, node_mask, d=None):
    if d is None:
        x = (torch.log(p_sigma / q_sigma) + 0.5 * (q_sigma ** 2 + (q_mu - p_mu) ** 2) / (p_sigma ** 2) - 0.5) * node_mask
        x = x.sum(-1).sum(-1)
    else:
        mu_norm2 = ((q_mu - p_mu) ** 2).sum(-1).sum(-1)
        x = d * torch.log(p_sigma / q_sigma) + 0.5 * (d * q_sigma**2 + mu_norm2) / (p_sigma**2) - 0.5 * d
    return x


def cdf_standard_gaussian(x):
    return 0.5 * (1. + torch.erf(x / math.sqrt(2)))
